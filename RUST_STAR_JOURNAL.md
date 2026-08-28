# Rust Star Work Journal

This is the operational source of truth for current state, evidence, blockers,
and next actions. Stable scope and contracts live in `RUST_STAR_PROJECT.md`.
User/hardware/account-dependent work lives in `RUST_STAR_MANUAL_TASKS.md`.

## Resume Protocol

Before changing code:

1. Read `RUST_STAR_PROJECT.md` and this file completely.
2. Run `git status --short --branch`, inspect `git log -1`, and verify remotes.
3. Confirm that the current-state summary still matches the repository.
4. Check whether the active oracle manifest and benchmark inputs are pinned.
5. Update this journal before handoff, context compaction, or the end of a
   meaningful session.

Journal entries are reverse chronological. Do not edit old entries to change
history; add a correction and update the current-state summary.

## Current State

- Phase: target-Mac bootstrap, quick `oracle-v1`, no-copy model mapping, the
  canonical differential-fixture envelope, complete layer 0, steady-state
  layer-0 execution, the first four-layer scheduler boundary, the minimum
  three-step position-advancing four-layer slice, and the generalized six-layer
  compression-schedule boundary are complete. A bounded position-127 replay
  first crossed the ratio-128 emissions for layers 3 and 5 with external
  activations; the integrated decoder now crosses the same boundary while
  committing the complete 128-token oracle transcript. The
  ordered position-advancing executor now extends through all 43 transformer
  layers and the exact post-transformer output boundary is complete through
  full-vocabulary logits and deterministic greedy selection. A separate
  four-position command remains an independent control. A position-127 command
  closes the feedback edge at every step, matches the final full-vocabulary
  logits and live layer-3/layer-5 ratio-128 rows, and has a diagnostic timed
  path with correctness readback excluded. A separate cold-prefill command now
  constructs position-0 raw/compressor state from the live prompt token, matches
  the complete prefill logits, and preserves the same position-127 exactness.
  Layers 0 through 42 execute in order under one Rust-owned Metal context with
  exact GPU-resident HC-state handoffs and one retained KV allocation per
  layer. Every even layer from 2 through 42 crosses its first ratio-4
  compressed-attention emission boundary at position 3, and the emitted rows
  remain visible at position 4; odd compressed layers retain ratio-128 state
  without emitting at this frontier.
  Layer 3 crosses from token-hash routing to biased top-k routing.
- Working branch: `agent/rust-star-bootstrap`.
- Branch base: upstream `antirez/ds4` commit
  `b0309611041655f4e45671cfd9c9886aff161406`.
- Local `main`: fast-forwarded to the same upstream commit.
- Fork `origin/main`: synchronized to the same upstream commit through the
  GitHub app.
- Oracle: `oracle-v1` is complete and independently verified. It binds source
  `b030961`, model SHA-256 `ca22ae2f...6261c0`, the target toolchain, 2K/32K
  full-vocabulary logits, correctness logs, and repeated performance evidence.
- Capture kit: `rust-star/capture_oracle_v1.py` prepares a privacy-filtered,
  checksummed result bundle from an isolated build of the pinned oracle source.
- Differential tooling: the initial bundle/full-logit format is stable and has
  cross-platform verification, exact C0 comparison, and drift diagnostics. The
  `rust-star-differential-fixture-v1` envelope now covers kernel,
  layer-segment, and decode-step scopes with strict tensor shape, path,
  finiteness, size, and hash verification.
- Benchmarking: `rust-star/BENCHMARK_PROTOCOL.md` fixes the v1 paired workload,
  eligibility, run ordering, aggregation, and capacity semantics. The paired
  raw/summary JSON contract and offline validator/aggregator are implemented
  and synthetically tested. A fresh-process DwarfStar adapter now normalizes
  one frontier while externally recording wall time and peak RSS. The
  checkpointed paired runner
  enforces warm-up, A/B and context ordering, explicit retries, and final
  validated export through an engine-neutral contract. Rust Star now has a
  fresh-process adapter and a raw `rust-star-engine-run-v1` producer for the
  exact 2K/128 path. Generation uses timing-only submission and validates the
  selected-token transcript afterward. The adapter independently rejects any
  raw run whose prefill/generation collection, transcript, or scheduler
  metadata is inconsistent. Native prefill now has a separate execution-only
  mode that retains its GPU state while omitting diagnostic fixture decoding,
  host boundary allocation, and boundary copies. The first five-pair target-Mac
  2K/128 development comparison completed without retries or invalid attempts.
  Rust Star's paired median was 0.7534x DwarfStar for steady decode, 0.3083x for
  complete generation, and 0.4133x for prefill. The first-token wall interval
  was 225.84x the oracle median pairwise ratio. Per-layer timing then traced the
  transition cliff to full-2K transient scratch retained across all 43 prefill
  layers, which exhausted unified-memory headroom and evicted early mapped
  weights before decode. Production prefill now pools same-role dead scratch,
  ping-pongs the cross-layer HC state, and releases prefill storage after the
  GPU state handoff while leaving diagnostic collection and persistent decoder
  state distinct. Two exact fresh-process 2K/128 development runs improved the
  first generated step to 833.639 and 732.984 ms, steady decode to 20.555 and
  20.789 tok/s, complete generation to 18.254 and 18.708 tok/s, and prefill to
  125.378 and 129.047 tok/s. Against the earlier five-pair medians this is about
  a 14.9x first-token improvement, 18.8% higher steady decode, 2.66x higher
  complete-generation throughput, and 73.3% higher prefill. The immutable
  `35ee7bb` candidate then completed a new five-pair exact comparison without
  retries or invalid attempts. Rust Star measured 20.031224379 tok/s steady,
  18.162539840 tok/s complete generation, 124.303074737 tok/s prefill, and
  708.321500 ms first-token latency. DwarfStar measured 23.56, 23.16, 181.83,
  and 47.741 ms respectively. The median within-pair ratios were 0.850831888x
  steady, 0.783881737x complete generation, 0.683350269x prefill, and
  14.531941455x first-token latency. Relative to the prior Rust medians, the
  validated candidate improved steady decode 14.5%, complete generation 2.58x,
  prefill 66.9%, and first-token latency 93.5%. The remaining paired steady gap
  was approximately 14.9%. A new explicitly ineligible per-layer profile found
  that steady transformer cost is distributed across the repeated layer
  schedule rather than concentrated in one pathological layer. Removing three
  unused diagnostic snapshots per layer, performing both RoPE transforms in
  place, and retaining a compute encoder across adjacent FFN stages reduced
  profiled steady transformer GPU time by 9.16% and raised profiled steady
  decode from 20.171 to 21.938 tok/s. Two fresh eligible runs reproduced 21.864
  and 21.769 tok/s with the exact 128-token transcript. Immutable commit
  `771b7c4` then completed five new exact pairs without a failure, retry, or
  invalid attempt. Rust Star measured 21.875065281 tok/s steady versus
  DwarfStar's 22.66, for a median within-pair ratio of 0.964571386x. The
  validated remaining steady gap was approximately 3.54%. Production-only GPU
  top-1 now avoids transferring and scanning the 129,280-logit row during timed
  greedy sampling. Immutable commit `4fc61f7` completed five new exact pairs
  without a failure, retry, or invalid attempt. Rust Star measured
  21.940665084 tok/s steady versus DwarfStar's 22.61, for a median within-pair
  ratio of 0.969791317x. The validated remaining steady gap is now approximately
  3.02%.
- Implementation: dependency-free Rust host scaffold under `rust-star/runtime/`
  strictly parses GGUF v3 directories, validates the Flash resident-Q2
  shape/recipe, and writes candidate full-logit artifacts. A macOS-only
  Rust/Objective-C/Metal boundary now owns a device, queue, pipeline, and
  correctness-checked shared-buffer dispatch probe. The complete Objective-C
  shim, optimized Rust build, shared-buffer probe, and real-model validator now
  pass on the M1 Ultra. The real GGUF exposed and fixed an incorrect Q8_0
  expectation for the F16 indexer Q projection. Rust now owns a read-only
  shared GGUF mmap; Metal wraps the page-aligned embedding tensor without a
  copy and runs the imported DwarfStar F16 embedding gather behind a bitwise
  CPU reference. The imported decode-time Q8_0 matvec now consumes a real
  layer-0 activation and the no-copy `blk.0.attn_q_a.weight` span, reproducing
  all 1,024 FP32 DwarfStar fixture outputs by bit pattern. The first connected
  chain now runs six imported operations in one command buffer from token 201's
  embedding through Q-Lora, with six mmap-backed model spans and bitwise checks
  at mixer, HC split, collapsed HC, learned attention norm, and Q-Lora.
  The connected path now covers all thirty layer-0 dispatches through Flash
  attention, output projection, hash routing, routed/shared experts, and final
  HC update. A narrow Rust `LayerExecutor` owns the Metal context, permanently
  binds it to one model mmap, and caches exact model views and activation
  buffers. It executes layers 0 through 3 in order; later layers omit
  embedding/repeat and consume the preceding layer's retained final HC Metal
  buffer directly. KV storage is keyed by layer identity rather than reusing
  one scratch allocation. All four layers use independently captured cache-row
  and differential fixtures and pass every retained boundary by FP32 bit
  pattern. Layer 2 derives its compressed RoPE base, scale, original context,
  and YaRN parameters from the model's compression schedule. A separate exact
  scheduler path commits the declared layer command buffers without inter-layer
  waits, retains layer-scoped activation lifetimes for deferred comparison, and
  performs one tail wait after the declared chain tail. The legacy three-layer
  chain declares layer 2, while the four-layer chain declares layer 3, so both
  remain independently validated scheduler controls. Layer 3 binds the model's
  256-value `exp_probs_b.bias` and uses biased top-k selection instead of the
  early layers' token-indexed hash tables. The four-layer steady-state harness
  now resolves all 100 model spans, decodes the four fixtures, and allocates
  host outputs once. Warmup and measured chains reuse those bindings; measured
  iterations collect only Metal timing metadata, followed by one exhaustive C0
  readback after the final sample. The same prepared executor now advances from
  token 201/position 1 through token 361/position 2 to token 1915/position 3
  without recreating its six layer-scoped KV caches. It verifies preserved
  FP16-rounded raw-cache history, grows each cache from two to four visible
  rows, derives RoPE/attention/router state from the explicit token and
  position, and hands off layer 5's final 16,384-element HC state. The original
  four-layer path remains an independently executed regression control. The
  decoder now owns recurrent attention-compressor state for layers 2 through 5
  plus indexer-compressor state for the even compressed layers 2 and 4. On the
  M1 Ultra it follows DwarfStar's
  separate paired F16 projection and one-row state-store path, retains the
  legacy concat/softmax/multiply/sum reduction for the first ratio-4 emission,
  applies learned norm, compressed RoPE, state shift, and FP8/indexer QAT, then
  appends the emitted layer-2 and layer-4 rows to attention. All retained
  boundaries match pinned position-1 and independently repeated position-2/3
  DwarfStar fixtures bit-for-bit. The generalized schedule applies ratio 4 plus
  indexer compression to even compressed layers and ratio 128 without indexer
  compression to odd compressed layers; at this frontier layers 2 and 4 emit
  while layers 3 and 5 only accumulate state. The new bounded ratio-128 replay
  independently owns 128 activation rows, projected KV/score state, and four
  no-copy model-weight views for each of layers 3 and 5. It submits all 128
  steps in one command buffer, performs the legacy single-row reduction only at
  position 127, and matches both 512-value DwarfStar emissions by FP32 bit
  pattern. Its inputs are externally captured `attn_norm` activations: it does
  not sample tokens and is not a complete decoder execution. The prepared
  executor's checked contiguous-chain contract now accepts eight layers. Layers
  6 and 7 retain independent raw caches and compressor states; layer 6 matches
  its first ratio-4 attention emission and indexer update at position 3, while
  layer 7 advances ratio-128 state without emitting. The four-layer and
  six-layer commands remain independently executed regression controls.
  The remaining layer-8–42 boundaries are represented by one checked fixture
  registry rather than pair-specific branches. The 43-layer command preserves
  all earlier controls, retains 43 raw caches and all layer-scoped compressor
  state, advances three exact decode positions, and hands off layer 42's final
  16,384-element HC state. The separate `decoder-output-probe` consumes that
  retained state without a host handoff and reproduces the four-value HC
  projection, HC weights, collapsed HC row, learned output norm, and all
  129,280 vocabulary logits at positions 1–3 by FP32 bit pattern. Lowest-ID
  argmax selects 361, 1915, and 262. The original command keeps those inputs
  external as a control. `closed-loop-decoder-probe` instead starts from token
  201 and feeds each selected token into the next position, then repeats the
  exact sequence without chained tensor collection. The timed intervals retain
  the required full-logit transfer and CPU argmax but exclude fixture readback
  and comparison. `position127-decoder-probe` extends the same ownership
  boundary through positions 1–127. With initial committed token 201 it
  reproduces all 128 oracle tokens, then compares the final 129,280 logits and
  persistent layer-3/layer-5 ratio-128 compressed rows bit-for-bit.
  `cold-prefill-decoder-probe` instead starts with empty Rust-owned cache and
  compressor storage, evaluates raw prompt token 36662 at position 0, matches
  all 129,280 prefill logits and selected token 201, and continues through the
  same 128-token transcript without loading captured initial state. A separate
  2K sequential initializer now owns a true 128-row raw-KV ring and
  context-sized compressed state. It matches two fresh DwarfStar one-token
  decode replays bit-for-bit at the final logits, but all 129,280 logits differ
  from two independently repeatable native batched-prefill captures while both
  select token 15342. The first native-batch arithmetic boundary is now isolated:
  the layer-0 hidden-combination and learned-normalization inputs are identical
  between schedules, while `blk.0.attn_q_a.weight` uses the M1 legacy
  `kernel_mul_mm_q8_0_f32` batch kernel and first diverges from the one-row
  matvec. A compact final-128-row fixture and live probe reproduce both kernels
  bit-for-bit and retain their expected 1,024-value final-row disagreement. A
  second compact final-32-row boundary now continues through the layer-0 KV
  projection, fused Q/KV learned RMSNorm, Q-B projection, and Q head
  RMSNorm/RoPE. Five no-copy model views and all 2,195,456 produced FP32 values
  match repeated native DwarfStar captures by bit pattern. The same final tile
  now starts from its 32 token IDs: embedding gather, four-stream HC repeat,
  plain RMSNorm, the legacy F16 batch mixer, and fused HC collapse/learned norm
  feed that Q/KV schedule in one ten-dispatch command buffer. Ten no-copy model
  views and all 2,457,600 produced FP32 values match repeated native DwarfStar
  captures by bit pattern. That boundary now continues through grouped
  attention output, FFN HC ingress and learned norm, the decomposed M1 router,
  fused IQ2_XXS routed gate/up plus weighted SwiGLU, Q2_K routed down and sum,
  the Q8_0 shared expert, and the additive FFN HC tail. The complete final-tile
  layer uses 43 dispatches and 25 no-copy model views; all 6,979,776 retained
  produced FP32 values and 192 selected expert IDs match repeated native
  DwarfStar captures exactly. A separate preserved boundary continues the same
  live command buffer into layer 1's HC ingress, learned attention norm, and
  Q-A projection. It uses 47 dispatches and 30 no-copy model views; the three
  new final tiles add 294,912 exact FP32 values without a host activation
  handoff. The 84-dispatch complete boundary now finishes layer 1's attention,
  routed/shared experts, and additive FFN HC tail with 49 no-copy views. The
  generalized executor is exact for two adjacent tiles spanning positions
  1984--2047. Its independent control loads a captured prefix for each tile;
  its persistent-context control instead retains both layers' live first-tile
  KV buffers and makes the final tile append to and consume that state. The
  final execution prefix is not assembled from the capture, while an explicit
  oracle comparison still guards both retained buffers before continuation.
  The same ownership contract now runs all 64 native tiles from position 0
  through 2047. It starts with empty KV buffers, consumes all canonical prompt
  tokens, and validates both accumulated layer prefixes before every append.
  This establishes exact complete-2K layer-0/layer-1 KV state. Every live
  layer-1 output tile now also feeds native layer-2 HC ingress, Q-A/KV
  projection, and fused learned norm. All 1,048,576 values in the repeated
  full-2K layer-2 `KVnorm` oracle match bit-for-bit, closing downstream
  validation of the non-final layer-1 outputs. The same 64-tile schedule now
  applies layer 2's compressed-attention RoPE and E4M3FN finalization, retains
  one live full-2K layer-2 raw-KV allocation, and matches every `KVrope` and
  `KVcur` value plus each accumulated prefix. The same empty-seed loop now owns
  both layer-2 ratio-4 compressors. It matches all 512 attention and indexer
  compressed rows and the four final recurrent state tensors bit-for-bit while
  preserving 65 no-copy model mappings per tile. The same persistent context
  now retains every normalized layer-2 query row and executes exact dense mixed
  attention over 2,048 raw plus 512 compressed rows. Q-B, compressed YaRN,
  FlashAttention, inverse RoPE, and both attention-output projections reproduce
  all 8,388,608 attention-output values bit-for-bit with four additional
  no-copy model mappings. The terminal command now continues through attention
  HC post-processing, the full token-hash routed/shared FFN, and the additive
  FFN HC update. Full 2K attention-HC and final-HC identities plus exact
  final-tile intermediates match the repeated oracle, completing native layers
  0, 1, and 2 at the 2K prompt boundary. The terminal command now carries that
  full final HC state directly through layer 3's HC attention ingress, learned
  norm, Q-Lora projection, Q/KV learned normalization, compressed RoPE, and FP8
  KV finalization. It retains all 2,048 layer-3 raw-KV rows plus the exact Q,
  HC, and attention-split state for continuation. All ten layer-3 boundaries
  match repeated DwarfStar captures exactly. The same terminal command now owns
  layer 3's full ratio-128 attention compressor. All 16 emitted FP8 rows and
  both 128x512 final recurrent-state tensors match two fresh DwarfStar processes
  bit-for-bit. It now also owns layer 3's dense mixed attention, inverse RoPE,
  Q8 output projections, and additive HC post. The full 2048x4096 attention
  output and final HC tile match repeated captures exactly, with the full HC
  identity checksum-pinned. The same command now owns layer 3's complete FFN:
  HC ingress, learned norm, biased top-6 selection, unbiased router-weight
  normalization, routed/shared experts, and the additive final HC update. All
  final-tile intermediates and the full final HC identity match two fresh
  DwarfStar processes exactly. The expanded command preserves 44/44 terminal
  no-copy mappings across 79 dispatches. That retained final HC now feeds layer
  4's HC attention ingress, Q/KV learned normalization, compressed RoPE, and
  FP8 KV finalization without a host activation upload. All ten final-tile
  boundaries match four fresh DwarfStar captures; the complete tensor
  identities are SHA-256 pinned. The expanded terminal command now also owns
  both layer-4 ratio-4 compressors. All 512 attention rows, all 512 indexer
  rows, and all four final recurrent-state tensors match two fresh DwarfStar
  processes bit-for-bit. Three additional no-copy views now drive layer 4's
  dense mixed attention over 2,048 raw plus 512 compressed rows, inverse RoPE,
  both Q8 output projections, and the additive HC post. The full 2048x4096
  attention output and final HC tile match two fresh DwarfStar processes
  exactly. Twelve final no-copy views now drive layer 4 through FFN HC ingress,
  learned norm, biased top-6 routing, routed/shared experts, and the additive
  final HC update. Every retained FFN boundary and the complete final HC
  identity match two fresh DwarfStar processes exactly. That final HC now feeds
  layer 5's HC attention ingress, learned norm, Q-A/KV projections, fused Q/KV
  normalization, Q-B, compressed RoPE, and FP8 KV finalization without a host
  activation upload. All ten layer-5 final-tile boundaries match four fresh
  DwarfStar processes exactly. That retained state now feeds layer 5's paired
  F16 compressor projections and exact full-batch ratio-128 schedule. All 16
  compressed rows and both final 128x512 recurrent-state tensors match two
  fresh DwarfStar processes bit-for-bit. Three additional no-copy views now
  drive layer 5's dense mixed attention over 2,048 raw plus 16 compressed rows,
  inverse RoPE, both Q8 output projections, and the additive attention HC post.
  The complete 2048x4096 attention output, diagnostic rows, final HC tile, and
  full HC identity match two fresh DwarfStar processes exactly. Twelve final
  no-copy views now carry that state through layer 5's FFN HC ingress, learned
  norm, biased top-6 routing, routed/shared experts, and additive final HC
  update. Every retained FFN boundary and the complete final HC identity match
  two fresh DwarfStar processes exactly. That retained final state now feeds
  layer 6's HC attention ingress, learned norm, Q-A/KV projections, fused Q/KV
  normalization, Q-B, compressed RoPE, and FP8 KV finalization. All ten new
  boundaries match four fresh DwarfStar captures exactly. The command preserves
  121/121 no-copy mappings across 236 dispatches after continuing through layer
  6's paired ratio-4 attention/indexer compressors. All 512 compressed rows and
  four final recurrent-state tensors match two fresh DwarfStar captures
  bit-for-bit. Three additional no-copy views now drive layer 6's dense mixed
  attention over 2,048 raw plus 512 compressed rows, inverse RoPE, both Q8
  output projections, and the additive attention HC post. The complete
  2048x4096 attention output, diagnostic rows, final HC tile, and full HC
  identity match two fresh DwarfStar processes exactly, establishing complete
  native layers 0–5 plus exact full-2K layer-6 attention HC post. Twelve final
  no-copy views now carry that state through layer 6's FFN HC ingress, learned
  norm, biased top-6 routing, routed/shared experts, and additive final HC
  update. Every retained FFN boundary and the complete final HC identity match
  two fresh DwarfStar processes exactly, establishing complete native layers
  0–6 at the 2K prompt boundary. The retained final state now continues through
  layer 7's complete Q/KV state, ratio-128 compressor, dense mixed attention,
  biased top-6 routed/shared FFN, and additive final HC update, then directly
  through layer 8's complete Q/KV state, paired ratio-4 attention/indexer
  compressors, dense mixed attention, routed/shared FFN, and final HC update,
  then through layer 9's complete ratio-128 path and layer 10's complete paired
  ratio-4 path, through layer 11's complete ratio-128 path, and then through
  layer 12's complete paired ratio-4 path, through layer 13's complete
  ratio-128 path, then through layer 14's complete paired ratio-4 path, and
  through layer 15's complete ratio-128 path, and then through layer 16's
  complete paired ratio-4 attention/indexer path, and then through layer 17's
  complete ratio-128 path, and then through layer 18's complete paired ratio-4
  attention/indexer path, and then through layer 19's complete ratio-128 path,
  then through layer 20's complete paired ratio-4 attention/indexer path, and
  then through layer 21's complete ratio-128 path, and then through layer 22's
  complete paired ratio-4 attention/indexer path, and then through layer 23's
  complete ratio-128 path, and then through layer 24's complete paired ratio-4
  attention/indexer path, then through layer 25's complete ratio-128 path,
  through layer 26's complete paired ratio-4 attention/indexer path, and then
  through layer 27's complete ratio-128 path, and then through layer 28's
  complete paired ratio-4 attention/indexer path, and then through layer 29's
  complete ratio-128 path, and then through layer 30's complete paired ratio-4
  attention/indexer path, then through layer 31's complete ratio-128 path, and
  then through layer 32's complete paired ratio-4 attention/indexer path, and
  then through layer 33's complete ratio-128 path, and then through layer 34's
  complete paired ratio-4 attention/indexer path, and then through layer 35's
  complete ratio-128 path, and then through layer 36's complete paired ratio-4
  attention/indexer path, then through layer 37's complete ratio-128 path, and
  then through layer 38's complete paired ratio-4 attention/indexer path, and
  then through layer 39's complete ratio-128 path, and then through layer 40's
  complete paired ratio-4 attention/indexer path, and then through layer 41's
  complete ratio-128 path, and then through layer 42's complete paired ratio-4
  attention/indexer path.
  All retained layer-7 through layer-42 boundaries, full
  attention outputs, compressor states, and full HC
  identities match fresh DwarfStar processes exactly, establishing complete
  native transformer layers 0–42 at the same prompt boundary with 1,216/1,216
  no-copy mappings across 2,372 terminal dispatches. The same persistent
  context now consumes the retained final layer-42 HC row through the exact
  five-dispatch output head. Its five additional model mappings preserve
  pointer identity, all 129,280 logits match the independently repeated
  DwarfStar batched-prefill frontier bit-for-bit, and lowest-ID argmax selects
  token 15342. This establishes complete native batched model prefill through
  logits with 2,377 dispatches and 1,221 no-copy mappings. A separate
  layer-2 position-2051 diagnostic now covers the complete ratio-4 sparse
  mechanism: F16 indexer projections, compressed RoPE, indexer QAT, direct
  scores, exact descending top-512 selection, the 12-way indexed mixed
  attention/reduction, and inverse RoPE. Twelve retained tensors from two fresh
  DwarfStar processes match bit-for-bit, and all three model spans preserve
  mmap pointer identity. The capture explicitly overrides the sparse threshold
  to 512; source inspection corrected the prior assumption that the default
  switched immediately after 512 rows. Pinned DwarfStar remains dense through
  1,024 compressed rows and first switches at 1,025. Two additional fresh
  production-default captures at position 4099 now match on every sparse
  boundary, including the two-block argsort merge required by 1,025 rows. The
  same schedule is wired into retained even-layer state, including same-step
  attention/indexer row commit and 35 no-copy mappings. A retained-state C0
  control now seeds the exact layer-2 state immediately before position 4099,
  executes the general retained layer path through its 1,025th compressed row,
  and matches 16 sparse-boundary tensors by bit pattern across 54 dispatches.
  It intentionally stops its correctness claim before the token-dependent FFN
  and does not claim preceding-layer execution. The retained scheduler now
  generalizes the same exact top-k algorithm beyond that first boundary. Its
  scratch allocation is fixed from context capacity, while each step derives
  the initial sort blocks, active work width, ping-pong offsets, merge count,
  and final top-512 dispatch from visible rows. A second seeded layer-2 control
  at position 8195 commits compressed row 2,049, runs three initial sort blocks
  and two merge passes over a 1,025-index work width, and matches the same 16
  boundaries across 55 dispatches with 35/35 mappings. A complete retained
  position-8195 decoder step now executes all 43 layers through the repeated
  sparse-merge branch, matches all 129,280 logits exactly, and selects token
  35597 across 1,813 transformer dispatches with 1,370/1,370 pointer matches.
  Full native model prefill and the eligible engine-measurement producer remain
  pending.
- Measurements: The exact complete native layers-0–35 full-2K command reported
  31809.591 ms wall / 31050.835 ms GPU in its focused correctness run across
  1,951 dispatches with 1,004/1,004 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 25011.323 ms wall / 24213.220 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–34 full-2K command reported
  14317.612 ms wall / 13546.455 ms GPU in its focused correctness run across
  1,904 dispatches with 976/976 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 19871.581 ms wall / 19730.470 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–33 full-2K command reported
  13126.804 ms wall / 12632.873 ms GPU in its focused correctness run across
  1,834 dispatches with 944/944 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 15919.880 ms wall / 15796.966 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–32 full-2K command reported
  14678.233 ms wall / 14571.577 ms GPU in its focused correctness run across
  1,787 dispatches with 916/916 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 13283.525 ms wall / 13179.070 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–31 full-2K command reported
  16492.572 ms wall / 15726.058 ms GPU in its focused correctness run across
  1,717 dispatches with 884/884 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 9087.549 ms wall / 8984.633 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–30 full-2K command reported
  12541.042 ms wall / 12450.093 ms GPU in its focused correctness run across
  1,670 dispatches with 856/856 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim. The complete
  target-Mac gate independently repeated the same exact boundary at
  8843.842 ms wall / 8740.028 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–29 full-2K command reported
  13595.215 ms wall / 13486.902 ms GPU focused and 8502.094 ms wall /
  8380.564 ms GPU in the complete target-Mac gate across 1,600 dispatches with
  824/824 no-copy model mappings.
  The isolated sparse indexed-attention diagnostic reported 18.864 ms wall /
  0.497625 ms GPU across 10 dispatches with 3/3 no-copy model mappings. Its wall
  interval includes command setup, synchronization, and exhaustive readback;
  neither value is a throughput claim.
  The production-default 1,025-row layer segment reported 25.162 ms wall /
  0.550000 ms GPU across 11 dispatches with 3/3 no-copy model mappings. The
  command also reran the 513-row diagnostic first. This is correctness evidence,
  not throughput. The complete target-Mac gate repeated the default boundary at
  22.303 ms wall / 0.547833 ms GPU with the same schedule and mapping counts.
  The focused retained-state boundary control reported 53.413 ms wall /
  29.894125 ms GPU across 54 dispatches with 35/35 pointer matches. It includes
  seed upload, synchronization, and exhaustive correctness readback and is not
  a throughput claim. The complete target-Mac gate repeated it at 58.511 ms
  wall / 30.369250 ms GPU with the same exact tensor, dispatch, and mapping
  counts.
  The focused retained multimerge control reported 61.399 ms wall /
  32.045250 ms GPU across 55 dispatches with 35/35 pointer matches. It includes
  seed upload, synchronization, and exhaustive correctness readback and is not
  a throughput claim. The complete target-Mac gate repeated it at 49.455 ms
  wall / 28.882250 ms GPU with identical tensor, schedule, and mapping counts.
  The complete target-Mac gate reported 2289.152 ms wall / 2161.306708 ms GPU
  with the same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims. The prior exact complete
  native layers-0/1/2/3/4/5/6/7 full-2K command reported 1950.645 ms wall /
  1864.381 ms GPU in its focused
  correctness run, across 313 dispatches with 164/164 no-copy model mappings.
  Its complete target-Mac gate reported 1915.398 ms wall / 1819.682 ms GPU with
  the same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims. The prior exact complete
  native layers-0/1/2/3/4/5/6 full-2K command reported 1709.387 ms wall /
  1620.434 ms GPU in its focused correctness run, across 266 dispatches with
  136/136 no-copy model mappings; its complete gate reported 1636.955 ms wall /
  1548.723 ms GPU.
  The prior exact complete native layers-0/1/2/3/4/5 plus layer-6
  dense-attention full-2K command reported 1545.043 ms wall / 1464.156 ms GPU
  in its focused correctness run, across 245 dispatches with 124/124 no-copy
  model mappings. The complete target-Mac gate reported 1635.054 ms wall /
  1454.353 ms GPU with the same schedule and mapping counts.
  The prior exact complete native layers-0/1/2/3/4/5 full-2K command
  reported 1358.984 ms wall / 1270.350 ms GPU in its focused correctness run
  and 1363.509 ms wall / 1272.876 ms GPU in the complete target-Mac gate,
  across 196 dispatches with 104/104 no-copy model mappings. The prior exact
  complete native layers-0/1/2/3/4 plus layer-5 dense
  mixed-attention full-2K command reported 1242.696 ms wall / 1152.157 ms GPU
  in its focused correctness run and 1253.568 ms wall / 1159.442 ms GPU in the
  complete target-Mac gate, across 175 dispatches with 92/92 no-copy model
  mappings. The prior layer-5 compressor checkpoint reported 1076.840 ms
  wall / 991.235 ms GPU focused and 1072.662 ms wall / 972.161 ms GPU in its
  complete gate, across 166 dispatches with 89/89 mappings. The prior layer-5
  Q/KV checkpoint reported 1085.161 ms wall / 994.834 ms GPU across 159
  dispatches and 85/85 mappings. The prior exact
  complete native layers-0/1/2/3/4 full-2K command
  reported 1079.281 ms wall / 991.837 ms GPU in its focused run and 1601.151 ms
  wall / 985.798 ms GPU in the complete target-Mac gate, across 149 dispatches
  with 76/76 no-copy model mappings. The prior attention-only checkpoint
  reported 934.193 ms wall / 833.479 ms GPU focused and 952.654 ms wall /
  851.603 ms GPU in the complete gate, across 128 dispatches with 64/64
  mappings. The prior paired-compressor checkpoint reported 740.517 ms wall /
  656.454 ms GPU focused and 750.548 ms wall / 655.250 ms GPU in the complete
  gate, across 119 dispatches with 61/61 mappings.
  The prior Q/KV-only checkpoint reported 1169.303 ms wall / 647.777 ms GPU in
  its focused run and 947.998 ms wall / 651.830 ms GPU in the complete
  target-Mac gate, across 89 dispatches with 53/53 mappings. The prior complete-layer-3
  checkpoint reported 675.511 ms wall / 592.734 ms GPU focused and 759.645 ms
  wall / 671.882 ms GPU in its complete gate. The layer-3 attention-only checkpoint
  reported 583.424 ms wall / 501.227 ms GPU focused and 569.588 ms wall /
  490.821 ms GPU in its complete gate. The prior ratio-128
  compressor checkpoint reported 494.638 ms wall / 396.686 ms GPU focused and
  483.530 ms wall / 399.101 ms in the complete target-Mac gate. The prior Q/KV
  command reported 482.230 ms wall / 402.931 ms GPU in its focused run and
  447.543 ms wall / 366.771 ms GPU in the complete target-Mac gate, across 42
  dispatches with 25/25 no-copy model mappings. These include full correctness
  readback and are not throughput claims. The prior layer-3-ingress control was 441.440 ms wall
  / 371.934 ms GPU, the layer-2-only control was 408.397 ms wall / 338.484 ms
  GPU, and the attention-only control was 196.069 ms wall / 169.211 ms GPU.
  Metal batching was 42.861x faster than synchronized submission
  in the retained M-002 probe. DwarfStar medians are 164.86 prefill / 19.90
  generation tok/s at 2K and 161.05 prefill / 17.36 generation tok/s at 32K.
  The first real-model kernel matched all 20,480 checked FP32 values; its final
  validation dispatch reported 0.020 ms GPU time and is not an inference-speed
  claim. The Q8_0 projection matched its 1,024-value decode fixture and reported
  0.031 ms GPU time in its completion gate, also not a throughput claim. The
  first native M1 batch boundary matched 131,072 Q-A outputs plus its 1,024-value
  sequential control; the complete gate reported 2.490 ms wall / 0.410 ms GPU
  for the isolated 128-row batch dispatch. All 1,024 final-row values differ
  between the native batch and sequential schedules with maximum absolute error
  `3.62396240234375e-05`; this is localization evidence, not prefill throughput.
  The extended final-32-row Q/KV setup matched all 2,195,456 produced FP32
  values; the complete gate reported 12.023 ms wall / 2.817 ms GPU for its
  five-dispatch isolated layer segment. This is correctness evidence, not
  prefill throughput. The first focused continuous token-to-Qcur run matched
  all 2,457,600 produced FP32 values, retained 10/10 mmap pointer identities,
  and reported 56.433 ms wall / 2.544 ms GPU. Its cold standalone wall time
  includes pipeline compilation and readback and is not a throughput claim. The
  complete gate repeated C0 exact at 55.079 ms wall / 2.561 ms GPU.
  The complete layer-0 final-tile boundary was C0 exact in its focused v5 run
  at 135.893 ms wall / 73.494 ms GPU with 25/25 no-copy model views. The full
  gate repeated it at 138.441 ms wall / 81.329 ms GPU. Both intervals include
  setup, synchronization, exhaustive readback, and comparison and are not
  prefill-throughput claims.
  The first exact layers-0/1 handoff run reported 125.125 ms wall / 69.515 ms
  GPU with 30/30 no-copy views; the full gate repeated it at 129.903 ms wall /
  78.876 ms GPU. These intervals have the same correctness-oriented scope and
  are not throughput measurements.
  The connected six-dispatch layer segment matched 9,264 retained FP32 values and
  reported 0.102 ms GPU time in the final gate; this remains correctness
  evidence rather than decoder throughput. The nine-dispatch Q/K projection
  setup matched another 34,816 FP32 values and reported 0.473 ms GPU time in
  its final gate, also not a throughput claim. The twelve-dispatch RoPE/cache
  gate matched 34,816 direct outputs plus the 512-value stored row, preserved
  two 512-value guard rows, and reported 0.949 ms GPU time; this is likewise
  correctness evidence rather than decoder throughput. Persistent layer-0
  execution measured 1.503 ms wall median (0.094 ms MAD) and 1.138 ms GPU
  median (0.033 ms MAD) across 30 bit-identical samples. The first continuous
  layers-0/1 correctness run reported 75.956/31.655 ms wall/GPU for cold layer
  0 and 27.276/26.313 ms for layer 1; those two-layer timings include setup and
  correctness readback and are not decoder-throughput claims. The canonical
  layers-0/1/2 gate reported 47.226/20.865 ms, 16.310/15.398 ms, and
  16.401/15.662 ms wall/GPU respectively; these intervals likewise include
  synchronization and correctness readback. Five alternating synchronized and
  chained samples measured 82.330 ms versus 80.175 ms median wall time, a
  1.0269x ratio or 2.62% reduction. Median summed command GPU time was 53.488 ms
  versus 52.458 ms, so the scheduler result remains a narrow diagnostic rather
  than a decoder-throughput claim. Over the four-layer slice, five alternating
  pairs measured 98.371 ms synchronized versus 97.216 ms chained median wall
  time, a 1.0119x ratio or 1.17% reduction. Median summed GPU intervals were
  68.128 and 67.986 ms; this is again a scheduler diagnostic, not a
  decoder-throughput claim. The prepared fixed-position four-layer replay
  measured 4.318 ms wall median (0.180 ms MAD) and 3.767 ms summed-GPU median
  (0.131 ms MAD) across 20 samples after five warmups, then passed one complete
  four-layer C0 collection. It is a steady-state execution microbenchmark, not
  token throughput. The first position-advancing run reported 103.106 ms wall
  and 70.862 ms summed GPU for its cold position-1 step, then 7.333 ms wall and
  6.347 ms summed GPU for position 2. Both steps were C0 exact across all four
  layers and cache histories; these remain correctness diagnostics rather than
  model-throughput measurements. The first exact position-3 compressor run
  reported 110.896/77.239 ms at the cold position-1 step, 8.863/8.123 ms at
  position 2, and 10.455/9.618 ms at position 3 (wall/summed GPU). It matched
  the emitted 512-value FP8 compressed KV row and every downstream layer
  boundary; this is also correctness evidence, not token throughput.
  The integrated six-layer gate reported 143.285/108.618 ms at the cold
  position-1 step, 14.098/13.178 ms at position 2, and 11.552/10.111 ms at
  position 3 (wall/summed GPU). It matched both ratio-4 emissions and every
  retained layer 0–5 boundary. These are correctness diagnostics, not token
  throughput measurements. In the final integrated gate, the bounded
  position-127 ratio-128 replay reported 32.461/8.236 ms wall/GPU for layer 3
  and 6.881/6.273 ms for layer 5. Both first emissions were C0 exact; setup,
  synchronization, and exhaustive comparison remain in scope, so these values
  are correctness diagnostics rather than inference-speed measurements. The
  integrated eight-layer gate reported 223.448/190.007 ms at its cold
  position-1 step, 16.989/16.442 ms at position 2, and 17.782/17.160 ms at
  position 3 (wall/summed GPU). It matched every retained layer 0–7 boundary
  and all three ratio-4 emissions. These values are also correctness
  diagnostics, not token-throughput measurements. The complete 43-layer gate
  reported 837.045/800.215 ms at position 1, 62.799/61.229 ms at position 2,
  and 53.374/49.691 ms at position 3 (wall/summed GPU). It matched every
  retained layer 0–42 boundary and all even-layer ratio-4 emissions through
  layer 42. These intervals also include correctness-oriented work and are not
  token-throughput measurements. The final output-head correctness gate
  reported 15.025/2.946 ms wall/GPU at cold position 1, 1.679/1.355 ms at
  position 2, and 1.691/1.359 ms at position 3. All 129,280 logits and four
  preceding output tensors were C0 exact at every step; these timings include
  synchronization/readback and are likewise not throughput claims. The first
  full-gate closed-loop diagnostic measured 54.610, 51.727, and 53.989 ms for
  its three committed steps after an exhaustive C0 preparation pass: 18.712
  tok/s complete and 18.919 tok/s over the final two steps. It includes command
  encoding, synchronized Metal execution, full logits transfer, and CPU argmax,
  but remains ineligible for a paired claim because it starts from captured
  state and generates only three tokens.
  The integrated position-127 diagnostic evaluates 127 closed-loop positions,
  reproduces the 128-token transcript, and reported 16.493 evaluated
  positions/s in the complete gate. Correctness readback occurs after timing.
  This is not paired throughput because the initial cache and compressor state
  are captured rather than produced by cold prefill.
  The first exact cold-state run paid model-residency costs and reported
  3379.018 ms for one-token prefill/first selection plus 0.729 evaluated decode
  positions/s.
  An immediate fresh-process repeat with resident weights reported 1130.755 ms
  prefill and 18.246 evaluated positions/s; the complete gate later reported
  1078.665 ms and 18.311 positions/s. All were bit-identical. None is a paired
  claim because one-token prefill is not a protocol frontier and the residency
  conditions differ. The first 2K sequential state run reported 18.603
  tokens/s over 110090.070 ms and was exact against the repeated DwarfStar
  decode replay. It is not prefill throughput: DwarfStar's native batched
  prefill is the required oracle and produced a different full-logit tensor.
  The first exact paired-compressor full-2K run reported 3624.704 ms summed
  wall and 3514.757 ms summed GPU time across its 64 correctness-oriented
  tiles. It retained 65/65 model mappings, used 118 dispatches on regular tiles
  and 122 on the final state refresh, and is not a prefill-throughput claim.
  The first complete-model native 2K prefill run reported 21478.015/21389.015
  ms wall/GPU for the terminal 43-layer transformer schedule and
  270.931/2.697 ms for its synchronized output-head correctness pass. Every
  one of the 129,280 batched-prefill logits was C0 exact and token 15342 was
  selected. These remain correctness timings, not an eligible engine
  measurement.
  The first exact native-prefill handoff used one 229-copy GPU blit command
  across all 43 layers, then matched all 2,052 post-prompt greedy selections
  through position 4099. Its first and final full-logit tensors were C0 exact,
  including all 21 even layers' first production sparse step at compressed row
  1,025. The correctness-oriented run reported 78.904 ms handoff wall time and
  16.563 synchronized positions/s; neither is a throughput claim.
- Manual handoff: `RUST_STAR_MANUAL_TASKS.md` records Actions approval, target
  compilation/model inspection, quick/extended oracle capture, and the deferred
  secure-access decision with exact evidence requirements.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: the connected GitHub app is installed on both `deathcoder` and
  `dion-labs`. Remote branch `agent/rust-star-bootstrap` is published and the
  validated checkpoints are pushed. Prefer the app for future remote writes
  from this environment.

## Immediate Next Actions

1. Use per-kernel or dispatch-family profiling to attribute the transformer
   portion of the remaining validated 3.02% steady gap without perturbing the
   eligible path. GPU top-1 removed about 0.20 ms/token of output-head wall time
   in two local A/B pairs, confirming that the output boundary is not the main
   residual. The current five-pair median is 21.940665084 tok/s for Rust Star
   versus 22.61 for DwarfStar, with a 0.969791317x median within-pair ratio.
2. Isolate the remaining first-token residency/scheduling cost: the new paired
   median is 871.961417 ms versus DwarfStar's 50.26 ms. Rust's internal median
   is 868.859875 ms transformer wall but only 67.742042 ms summed transformer
   GPU, while the output head is 3.281083/2.755208 ms wall/GPU. Do not move a
   second weight warm outside the declared interval merely to improve the
   metric.
3. Preserve the exact 2K-to-position-4099 native handoff, complete retained
   position-8195 decoder step, isolated
   513/1,025-row probes, and retained-state row-1,025/2,049 controls as
   independent sparse regressions.
4. Preserve the four-, six-, eight-, 43-layer, explicit decoder-output, and
   closed-loop diagnostic commands as independently executed controls.
5. Run the extended 2K--1M frontier capture when the Mac can be dedicated to a
   long benchmark; preserve any 512K/1M capacity failure as evidence.
6. Run or approve the fork's GitHub Actions workflow and retain its URL.

## Entries

### 2026-08-28 — Production sampling moves lowest-ID argmax onto Metal

Objective:

- Test whether Rust Star's full 129,280-logit host copy and checked CPU scan
  account for the residual 3.54% steady gap, after confirming that DwarfStar's
  greedy path performs top-1 selection on the GPU.

Actions and evidence:

- Added a finite-checked, lowest-token-ID Metal reduction to the existing
  output-head command. Production timing now reads an eight-byte token/validity
  result; full-logit and intermediate readback remain unchanged in every C0
  diagnostic.
- The complete 2K/128 engine path matched the exact 128-token oracle transcript
  in every run. All 288 optimized Rust tests passed, including the unchanged
  full-logit fixture and stable-report controls.
- Two alternating local development pairs compared the immutable `771b7c4`
  executable with the new path. Pair ratios were 1.001781148x and 1.007805722x
  steady throughput. Old/new medians were 21.793512982 and 21.897939975 tok/s,
  a provisional 1.004793435x ratio.
- Output-head wall time fell from 2.487554 to 2.262153 ms/token in the first
  pair and from 2.506579 to 2.332198 ms/token in the reversed-order pair. The
  single-threadgroup reduction added about 0.057 ms/token of GPU work, leaving
  a median wall saving of about 0.20 ms/token.
- One earlier run immediately after the large optimized link was invalid for
  performance interpretation: transformer wall time rose to roughly 1.50
  seconds/token under page pressure. Its transcript was exact, and an immediate
  clean rerun restored normal 21.861 tok/s behavior. It is retained only as
  failure evidence under `rust-star/.work/top1-output-head-01/`.
- Valid development evidence is retained under
  `rust-star/.work/top1-output-head-02/` and
  `rust-star/.work/top1-output-head-ab/`.
- Published source commit `4fc61f789e3e1df726833c32f7b1be0446c43b2a`
  with tree `53872576e334aa2f9a76c561142d5ec85320fa4f`, then copied its
  optimized executable to an immutable local candidate path. Executable
  SHA-256 is
  `22eb2bab4f6a52f34c73fad8b018695c451d3120e6900511304f4504808b3ea8`.
- Froze plan SHA-256
  `1511d56f177be34ec3aa6ee48f21b1e9fed3a04d9917c35ac5d9dec3f832c88a`.
  Both warmups and all five `AB`, `BA`, `AB`, `BA`, `AB` pairs passed with no
  adapter failures, retries, invalid attempts, capacity failures, or observed
  thermal events.
- Rust Star steady decode measured median 21.940665084 tok/s (MAD
  0.067226655, range 21.639385646--22.014262891) versus DwarfStar 22.61 tok/s
  (MAD 0.04, range 22.57--22.71). The median within-pair ratio was
  0.969791317x (MAD 0.003578508, range 0.958767640--0.973369825), leaving a
  validated 3.02% steady gap.
- Complete generation measured Rust 19.218353297 versus DwarfStar 22.06 tok/s,
  for a 0.872293133x median pairwise ratio. Prefill measured 124.056813477
  versus 181.19 tok/s, for 0.685070803x. First-token latency measured
  871.961417 versus 50.26 ms, for a 17.533320056x median pairwise ratio. This
  optimization addresses only steady sampling; it makes no prefill or
  first-token claim.
- Independent aggregation reproduced the byte-identical summary. Raw SHA-256
  is `aaac87bc1270a1be0a8767e6a1c8c8dacebf1393ae20da1a2a96ede11e5d6cf9`;
  summary SHA-256 is
  `ada1073a90e21168be375642bd0ca6943d76bcfac83fad75745db52a25b22626`.
  The local result-manifest SHA-256 is
  `e023ba3b82a8e6fdd98db40ad9ede7a9906f524bb2daaa1f8296ec142ac1f65c`.
  Full private-path evidence is under
  `rust-star/.work/paired-2k-128-4fc61f7/` and intentionally untracked.

Validation:

- Optimized macOS build and all 288 Rust tests.
- All 68 Python artifact/adapter/paired-runner tests.
- Exact retained position-8195 control across 43 layer HC states and all
  129,280 final logits; selected token 35597. Report SHA-256 is
  `165443d86ddf8187dc4bd172ca55fae17944ce397da6280eaeecb4d4258be1a9`.
- Candidate executable SHA-256 is
  `22eb2bab4f6a52f34c73fad8b018695c451d3120e6900511304f4504808b3ea8`.
- `cargo fmt --check` and `git diff --check`.

Decision and next step:

- Keep GPU top-1 as the new validated production path aligned with DwarfStar's
  greedy boundary. The five-pair steady median improved 0.30% over `771b7c4`,
  and the paired ratio improved 0.54%, but the gain is small and does not
  motivate broad unchecked Rust memory access.
- Continue attribution inside repeated transformer dispatch families; they now
  account for essentially all of the remaining 3.02% steady gap.

### 2026-08-28 — Cross-layer command-buffer grouping rejected

Objective:

- Determine whether the remaining 3.54% steady gap comes from Rust Star's 43
  transformer command buffers per generated token versus DwarfStar's larger
  batch submission.

Experiment and evidence:

- Implemented an internal production-only command-buffer grouping contract.
  Exact diagnostic paths retained one buffer per layer, while the seeded
  position-8195 complete-decoder control exercised each grouped production
  schedule. Every tested grouping remained C0 across all 43 final HC states,
  all 129,280 logits, and selected token 35597.
- One transformer command buffer was decisively slower: its eligible exact run
  measured 19.317714414 steady tok/s. Steady transformer GPU time remained
  about 40.384 ms/token, but transformer wall time rose to 49.261 ms/token
  because the GPU could not start until Rust finished encoding the complete
  1,813-dispatch transformer graph.
- Four-layer groups, eleven transformer buffers plus the output head, measured
  21.674 steady tok/s and also failed to beat the validated baseline.
- Two-layer groups, 22 transformer buffers plus the output head, measured
  22.166830346, 21.910702064, and 21.912 steady tok/s in three fresh eligible
  processes. The median is only about 0.17% above the validated 21.875065281
  tok/s baseline and lies inside observed run variation; it does not justify a
  new scheduler contract or paired run.
- Private evidence is retained under
  `rust-star/.work/single-transformer-command-*`,
  `rust-star/.work/four-layer-command-groups-*`, and
  `rust-star/.work/two-layer-command-groups-*`.

Decision and next step:

- Rejected all command-buffer grouping variants and restored the validated
  43-buffer production scheduler. Queue-ordered per-layer commits are useful:
  they overlap GPU execution of early layers with CPU encoding of later ones.
- The residual steady gap is not command-buffer commit overhead. Continue with
  kernel/dispatch-family attribution inside the repeated layer GPU work.

### 2026-08-28 — Production scheduling reaches 96.46% of DwarfStar in five exact pairs

Objective:

- Freeze the production no-snapshot schedule and determine whether its 8.76%
  profile gain survives the predeclared alternating-order paired protocol.

Actions and evidence:

- Published source commit `771b7c4ccb5beabbeb145e09a7f5fe73c582f20d`
  with tree `750c2f65da27de5afd98f3c8647f9ba9eba63dfb`, then copied its optimized
  executable to an immutable local candidate path. Executable SHA-256 is
  `56c252ab8def1b0ce71580757accb27ff3a21dd0c96cbff0fffbf4191f71e05b`.
- Froze plan SHA-256
  `e0ab758fdd751e7ef69fc9313f40745defade85585021008a7a68a89520ea51e`.
  Before the run the Mac was on AC, reported no recorded thermal/performance
  warning, had 94% free memory, zero throttled pages, and no model process.
- Both full-length warm-ups passed. Five pairs completed in the declared `AB`,
  `BA`, `AB`, `BA`, `AB` order with zero adapter failures, retries, invalid
  attempts, capacity failures, or observed thermal events. Afterward the Mac
  still reported 94% free memory, zero throttled pages, and no recorded thermal
  or performance warning.
- Rust Star steady decode measured median 21.875065281 tok/s (MAD
  0.012764947, range 21.789438485--21.887830228) versus DwarfStar 22.66 tok/s
  (MAD 0.05, range 22.59--22.74). The within-pair median was 0.964571386x
  (MAD 0.000363122, range 0.962476742--0.968059718). The remaining validated
  steady gap is 3.54%, down from 14.9% at `35ee7bb`.
- Complete generation measured Rust 19.343260789 tok/s versus DwarfStar 22.1,
  for a 0.874572570x median pairwise ratio. Prefill measured 123.187716755
  versus 180.01 tok/s, for 0.684338185x. First-token latency measured
  812.705917 versus 50.857 ms, for 15.505437619x.
- Relative to `35ee7bb`, the Rust median improved 9.20% in steady decode and
  6.50% in complete generation. Prefill changed -0.90%, while first-token
  latency worsened 14.74%; this optimization does not address either frontier.
- Independent aggregation reproduced the byte-identical summary. Raw SHA-256
  is `ac80b093057f15f0fa5ecf8a96c09fd417a0ee1050ca97176b083c00ed1af0a9`;
  summary SHA-256 is
  `1f4833ffa653921fafb639d45026efe7d218ab6277201f256ed7f55ba985c9d1`.
  The local result-manifest SHA-256 is
  `8ff7475a10ccbcee7f94372efa736304efd828905dfb1d46f34e3a310a10d70a`.
  Full private-path evidence is under
  `rust-star/.work/paired-2k-128-771b7c4/` and intentionally untracked.

Decision and next step:

- Accept 96.46% of DwarfStar as the new validated steady baseline. Rust Star
  has not yet beaten DwarfStar, but the residual is now small enough for
  dispatch-family/kernel attribution rather than broad host-side optimization.
- Profile the remaining repeated transformer GPU work without adding reads or
  waits to `engine-measure`; keep first-token residency and prefill as separate
  problems.

### 2026-08-28 — Production decode drops unused snapshots and redundant encoder boundaries

Objective:

- Attribute the remaining 14.9% paired steady-decode gap and keep the best
  production optimization only if the retained complete-decoder C0 control
  remains bit-identical.

Changes and evidence:

- Added `engine-profile`, an explicitly paired-ineligible form of the exact
  2K/128 engine workload. It reads the already completed 43 layer command
  buffers' GPU timestamps after each generated-token interval and emits the
  summed steady per-layer values. Normal `engine-measure` remains separate,
  emits `null` for the profile array, records collection as false, and remains
  paired eligible.
- Baseline profile evidence under
  `rust-star/.work/steady-layer-profile-01/` has manifest SHA-256
  `524aa791a3e853fa9049cf718207662fd3306a34a6862490d961be8ea28e0ecd`.
  It measured 20.171179318 steady tok/s, 46.960839 ms transformer wall and
  44.164538 ms transformer GPU per steady token. The hottest single layer was
  only 1.128632 ms/token or 2.56% of transformer GPU time; ratio-4 even layers
  accounted for 51.68% and ratio-128 odd layers for 44.19%. The remaining cost
  was therefore repeated schedule overhead/work, not one anomalous layer.
- Source comparison with pinned DwarfStar showed that its batch path reuses a
  compute encoder, while Rust Star's timing-only production layer still kept
  three probe-only snapshots and repeatedly ended/recreated encoders. Added an
  internal output-collection contract to the Rust/Objective-C boundary. Exact
  tensor probes preserve the old observable schedule. Production decode skips
  the Q/pre-RoPE-KV, post-RoPE-KV, and attention pre-inverse-RoPE snapshots,
  executes Q and inverse RoPE in place, and keeps adjacent FFN dispatches in
  one encoder. Required persistent compressed-cache commits still use blits and
  retain their dependency boundaries.
- Optimized profile evidence under
  `rust-star/.work/steady-layer-profile-02-inplace/` has manifest SHA-256
  `b62c868d6e46179259f619564d17084a711ec9f0c0103abfc7da613b35ec8b11`.
  It measured 21.937837562 steady tok/s, up 8.76%; transformer wall fell 8.29%
  to 43.066214 ms/token and transformer GPU fell 9.16% to 40.121048 ms/token.
  Prefill was effectively unchanged. The first-token sample worsened from
  741.225 to 795.800 ms, so this checkpoint makes no first-token improvement
  claim.
- The optimized production schedule passed the seeded position-8195 retained
  decoder control: every one of 43 16,384-float layer HC outputs and all
  129,280 logits matched DwarfStar by FP32 bit pattern, and the exact selected
  token remained 35597. The report SHA-256 is
  `2202486531c149d2666515334a357a317e03d7ebf1bbc7ee1823550ba20e4c02`
  under `rust-star/.work/production-schedule-c0-01/`.
- Two independent non-profiled `engine-measure` processes remained paired
  eligible, matched transcript checksum `17615242442502606640`, and measured
  21.863585930 and 21.769 steady tok/s. Their complete-generation rates were
  19.376402772 and 19.363778103 tok/s; prefill measured 123.294150291 and
  123.200729385 tok/s. Evidence is under
  `rust-star/.work/production-schedule-measure-01/` and intentionally untracked.

Validation:

- Optimized macOS release build.
- Retained position-8195 complete-decoder C0 control across all layer HC states
  and full logits.
- Two exact fresh-process eligible 2K/128 engine measurements.
- `cargo fmt --check` and all 288 Rust tests.
- All 68 Python artifact/adapter/paired-runner tests.
- `git diff --check` before publication.

Decision and next step:

- Keep the production-only schedule optimization. It applies the useful
  Polars-style lesson—remove redundant materialization and preserve a fixed
  execution plan—at the GPU scheduling boundary where profiling found the
  cost; no broad or unjustified `unsafe` Rust was added.
- Freeze the new commit and run a fresh five-pair alternating-order comparison.
  Until then the 21.8 tok/s development result is evidence, not a replacement
  for the last validated paired ratio.

### 2026-08-28 — Scratch-residency gain reproduces in five exact pairs

Objective:

- Freeze `35ee7bb` and determine whether the diagnostic scratch-residency gain
  survives the predeclared alternating-order paired protocol.

Changes and evidence:

- Rebuilt the optimized runtime from committed source. Candidate executable
  SHA-256 is
  `1f401310cdde2d15a127a4833269e09f4f67f1c4cb85933ebc7a2cce0201968d`;
  source tree is `59c18a01cfcb62a523aaa6baef470fb29e6ad114`.
- Captured a fresh privacy-filtered host manifest with AC power, no recorded
  thermal/performance warning, 95% free memory, zero throttled pages, and no
  model process. Froze plan SHA-256
  `ed69618ada4d5a830ed327de48f5229140f36a72843c0f3638bea79d627aef65`.
- Both full-length warm-ups passed. All five 2K/128 pairs completed in the
  declared `AB`, `BA`, `AB`, `BA`, `AB` order with zero failures, retries,
  invalid attempts, capacity failures, or observed thermal events. Post-run
  memory returned to 93% free with zero throttled pages.
- Rust Star steady decode measured median 20.031224379 tok/s (MAD
  0.018443385, range 19.987054309--20.050703952) versus DwarfStar 23.56 tok/s
  (MAD 0.06, range 22.54--23.62). The within-pair candidate/oracle median was
  0.850831888x (MAD 0.003550728, range 0.847281160--0.888696734).
- Complete generation measured Rust 18.162539840 tok/s versus DwarfStar 23.16
  tok/s. The within-pair median was 0.783881737x (range
  0.779761425--0.827919048).
- Prefill measured Rust 124.303074737 tok/s versus DwarfStar 181.83 tok/s. The
  within-pair median was 0.683350269x (range 0.681938272--0.685419985).
- First-token latency measured Rust 708.321500 ms versus DwarfStar 47.741 ms.
  The within-pair median ratio was 14.531941455x (range
  12.219232661--15.366224835).
- Relative to the prior five-pair Rust medians, scratch reuse improved steady
  decode 14.5%, complete generation 2.58x, prefill 66.9%, and first-token
  latency 93.5%. The paired steady ratio improved from 0.753377393x to
  0.850831888x. Rust Star has not yet beaten DwarfStar.
- Finalization rehashed every measurement. Independent aggregation reproduced
  the byte-identical summary. Raw SHA-256 is
  `139561e3dd048a0a5d504e841aa8cd09e254dc22b42e39a5c063b88f682fa970`;
  summary SHA-256 is
  `b0eb0563e5bf8ce0570650dc75c955c065b9259e3a1a41ad5cdf018670d329df`.
  Complete private-path evidence remains under
  `rust-star/.work/paired-2k-128-35ee7bb/` and is intentionally untracked.
- Internal timing medians place steady transformer execution at 47.247 ms wall
  and 44.693 ms summed GPU per token. The output head costs 2.666 ms wall and
  1.386 ms GPU per token. The first generated step is 705.921/688.398 ms in the
  transformer but only 2.875/2.190 ms in the output head, so both remaining
  gaps are transformer/device-residency problems rather than output sampling.

Performance-engineering note:

- The suggested animal-branded Rust project was likely Polars. Its transferable
  advantages are columnar/contiguous data, zero-copy Arrow views, fixed query
  plans, SIMD, parallelism, and eliminating allocations and redundant work;
  `unsafe` supports a few proven low-level primitives but is not itself the
  source of speed. Rust Star should continue applying the analogous zero-copy,
  fixed-shape, buffer-lifetime, and scheduled-work principles. Test
  `-C target-cpu=native` and unchecked host loops only as isolated profiled
  candidates. Keep unsafe surfaces narrow, document their caller invariants,
  and retain a safe/C0 differential control; current timing shows Metal GPU
  work dominates, so broad unsafe Rust changes would not close the main gap.

Decision and next step:

- Keep and publish the scratch-residency checkpoint and its paired evidence.
  Next add timing-only steady per-layer/kernel attribution, then optimize the
  largest Metal schedule or memory-traffic contributor. Separately trace the
  residual first-token weight-residency cliff without shifting warm-up work out
  of a declared inference interval.

### 2026-08-28 — Timing-only prefill scratch reuse removes decoder residency cliff

Objective:

- Explain and remove Rust Star's 10.95-second first generated step without
  changing arithmetic, weakening C0, or instrumenting the timed paired runs.

Changes and evidence:

- Added DwarfStar-equivalent mapped-weight warming before internal inference
  timers: `POSIX_MADV_WILLNEED` plus one volatile byte touch per tensor-data
  page. The raw engine report now records warm bytes, pages, checksum, and wall
  time so residency policy is explicit and independently auditable.
- Added untimed diagnostic breakdowns for decoder preparation, first and steady
  transformer wall/GPU time, output-head wall/GPU time, and the first token's
  43 completed per-layer command-buffer GPU intervals. Reading those timestamps
  submits no work and occurs after the first-token stopwatch.
- The diagnostic run showed early decoder layers taking roughly 676--701 ms
  each while late layers took roughly 23--29 ms. Native timing-only prefill had
  encoded the entire 43-layer schedule into one command buffer while retaining
  distinct multi-row scratch for every layer. That working set consumed the
  remaining 128 GB unified-memory headroom and evicted early mapped weights in
  layer order; the first decode then faulted them back in sequentially.
- Production native prefill now reuses transient buffers by semantic role for
  layers 3--42. The final HC buffer uses two parity slots because the next layer
  consumes it. Raw KV, compressed attention/indexer rows, and their score/KV
  states remain unique per layer. Diagnostic collection still allocates every
  layer boundary independently. After the decoder-state blit completes, the
  prefill properties and pooled activation-cache entries are released.
- Fresh-process run `rust-star/.work/rust-star-ctx-2048-warm-04` remained exact
  and reported 125.377945557 prefill tok/s, 833.638750 ms first token,
  20.554818230 steady tok/s, and 18.253799463 complete-generation tok/s. Its
  first-token per-layer median was 20.472 ms and maximum 73.675 ms.
- Fresh-process repeat `rust-star/.work/rust-star-ctx-2048-warm-05` reproduced
  the exact 128-token checksum `17615242442502606640` and reported
  129.046988121 prefill tok/s, 732.983916 ms first token, 20.788667769 steady
  tok/s, and 18.707757287 complete-generation tok/s. Its first-token per-layer
  median was 20.113 ms and maximum 28.771 ms.
- Relative to the earlier five-pair Rust medians, the repeat is about 14.9x
  faster on first-token latency, 18.8% faster on steady decode, 2.66x faster on
  complete generation, and 73.3% faster on prefill. This is diagnostic evidence,
  not a paired headline result. Rust Star is now approximately 9.4% behind the
  earlier DwarfStar steady median of 22.95 tok/s.
- The full Mac regression resumed after an interruption and passed every
  remaining exact model gate: 2K live-KV, layer-2 KV norm/state, compressor,
  exact native 2K prefill across all 1,216 no-copy mappings, full 43-layer
  decode/output head, four-token closed loop, position 127, native prefill
  handoff through position 4099, sparse/multimerge controls, and retained
  position 8195. The complete Rust suite passed 287 tests; the Python suite
  passed all 68 tests; formatting, optimized Mac compilation, and the runtime
  model gates are green.

Decision and next step:

- Keep the C0-safe scratch lifetime optimization. Commit and freeze this
  candidate, then rerun the unchanged five-pair 2K/128 protocol. If reproduced,
  optimize the remaining steady transformer GPU/command path; the output head
  is no longer the dominant gap.

### 2026-08-27 — First checkpointed C0 paired 2K/128 comparison complete

Objective:

- Replace single-run bring-up numbers with the predeclared five-pair,
  alternating-order development comparison against the pinned DwarfStar oracle.

Changes and evidence:

- Corrected the paired runner's warm-up to use the plan's declared generation
  length. The prior hardcoded eight-token warm-up was incompatible with the
  intentionally narrow exact 2K/128 candidate. The protocol requires an untimed
  smallest-context warm-up but does not require a shorter generation. The
  checkpoint/resume test now verifies both warm-up artifacts use 128 tokens.
- Committed that runner-only change as `983d5f3`, rebuilt the unchanged Rust
  runtime release binary, and passed all 68 Python tests. The candidate
  executable SHA-256 remained
  `0754a2e3edf8563002b4d83d8642f1aef85bc7426f1d2dd2109a95ecc1975822`.
- Froze an immutable local plan with SHA-256
  `4eeee3a90e830ccfc446e6762a8ad55a34c1792b3d484b2ef6f702f5665b81da`.
  It binds source commits/trees, exact executable/model/prompt hashes, explicit
  build/runtime configurations, the privacy-filtered host manifest, and the C0
  correctness manifest.
- Both full-length warm-ups passed. All five timed 2K/128 pairs completed in
  alternating `AB`, `BA`, `AB`, `BA`, `AB` order with zero failures, retries,
  invalid attempts, capacity events, or recorded macOS thermal/performance
  warnings.
- Rust Star steady decode had median 17.493423061 tok/s (MAD 0.425360410,
  range 17.068062651--19.317236451) versus DwarfStar 22.95 tok/s (MAD 0.27,
  range 22.64--24.03). The within-pair candidate/oracle median was 0.753377393x
  (MAD 0.005669741, range 0.747707652--0.803880002).
- Complete generation, which includes the first generated step, measured a
  paired median 0.308332588x (range 0.298489542--0.319406508). Rust Star's
  median was 7.041875502 tok/s versus DwarfStar's 22.60 tok/s.
- Prefill measured a paired median 0.413274711x (range
  0.375442069--0.493098704). Rust Star's raw median was 74.480368506 tok/s
  versus DwarfStar's 178.29 tok/s.
- Rust Star's first generated step took a 10,952.916750 ms median versus
  DwarfStar's 49.088 ms. The within-pair latency ratio was 225.843754304x
  (range 210.176285188--237.544243664), explaining why its complete-generation
  ratio is far below its steady-state ratio.
- Finalization rehashed every measurement. Independent aggregation reproduced
  the byte-identical summary. Raw SHA-256 is
  `b8daf73a1ef556d9181e0100fe9dadec0c6d65c14d798121440d1e5548571997`;
  summary SHA-256 is
  `7ebc757a361584de75ae452ce757a4d683a0b5343f709948ad4b40c6c7ed5c78`.
  Complete private-path evidence remains under
  `rust-star/.work/paired-2k-128-983d5f3/` and is intentionally untracked.

Decision:

- This is valid C0 development evidence at 2K, not the protocol's publishable
  256K primary claim. Rust Star is currently slower at every user-visible
  metric. Preserve the negative result and optimize from its measured shape:
  first remove the cold decoder-transition penalty, then improve native prefill
  and the remaining roughly 25% steady-decode gap.

Next:

- Add a correctness-neutral diagnostic around the first post-prefill decoder
  step to distinguish resource/pipeline warm-up and weight residency from real
  transformer/output-head execution. Keep the paired interval unchanged and
  rerun the five-pair plan only after a measured fix.

### 2026-08-27 — Timing-only native prefill reached an eligible 2K/128 measurement

Objective:

- Remove exhaustive prefill boundary collection from measured execution while
  preserving the exact complete-model C0 command unchanged.

Changes and evidence:

- Added an explicit `collect_outputs` boundary to both native prefill Metal
  ABIs. Diagnostic callers still validate every output pointer and copy every
  retained tensor; timing callers execute the same command schedules while
  retaining only GPU-owned decoder state.
- The Rust timing path does not decode diagnostic output fixtures or allocate
  their host tensors. It transfers only final output-head logits for mandatory
  lowest-ID argmax and validates the complete selected-token transcript after
  all generation intervals.
- The unchanged `prefill-layers012-attention-loop-probe` passed on the M1 Ultra:
  2,372 transformer dispatches, 1,216/1,216 no-copy model ranges, 17,772.379 ms
  wall time, exact full logits, and selected token 15,342. Its JSON SHA-256 is
  `df1c7eedb4e1a7079c470d1575618088239aa113b1ad83e9c1a02e413f4f7577`.
- The first timing attempt failed safely before Metal work on one remaining
  diagnostic raw-cache slice. The guarded retry passed `measure_ruststar.py`
  with both collection flags false, 44 command buffers and two waits per
  generated token, and the exact 128-token oracle transcript.
- The clean eligible single-run artifact reported 90.668305877 prefill tok/s,
  7.775007935 generation tok/s including the first step, and 18.406557599
  steady tok/s. Intervals were 22,587.826917 ms prefill, 9,563.290792 ms first
  generation step, and 6,899.714915 ms for the remaining 127 tokens. The normalized
  measurement SHA-256 is
  `6fb9f647ff6926b784f2dd377672f29577181afec8792c3621205bbe8714bca6`.
- The adapter recorded 73,873.229625 ms complete-process wall time and
  121,602,048 bytes child peak RSS. This is the adapter's host RSS field, not a
  claim about total Metal/unified-memory residency.
- The complete host-runtime gate passed: formatting, 287 Rust tests, optimized
  macOS build, the M1 Ultra dispatch probe, 68 Python tests, every pinned
  differential fixture, the Rust/Python artifact smoke test, all native prefill
  controls, the full 2K-to-position-4099 handoff, retained sparse controls, and
  the steady-state diagnostic benches.

Decision:

- The raw producer may now set `paired_protocol_eligible: true`; the adapter
  still fails closed if either collection flag, transcript, scheduler metadata,
  or blocker field is inconsistent. The exact C0 command remains an independent
  regression control and is not replaced by transcript validation.

Next:

- Run the checkpointed alternating-order DwarfStar/Rust Star A/B plan at
  2K/128. Treat this single Rust Star run as bring-up evidence, not a comparative
  performance conclusion.

### 2026-08-27 — Rust Star measurement boundary added; prefill collection isolated as blocker

Objective:

- Connect the exact native-prefill/decoder path to the engine-neutral paired
  process contract without reporting correctness-oriented intervals as speed.

Changes and evidence:

- Added `engine-measure` and stable `rust-star-engine-run-v1` output for the
  exact 2,048-token prefill plus 128 committed greedy tokens.
- The generation loop prepares model bindings outside timing, includes command
  encoding, 44 synchronized command buffers, two waits, full output-head
  sampling, and token commitment per step, and compares all selected tokens
  with the pinned oracle only after timing.
- Added `measure_ruststar.py` and its library. The fresh-process adapter records
  wall time and child peak RSS, sanitizes logs, checksums evidence, normalizes
  rates, and refuses inconsistent eligibility, collection, transcript, or
  scheduler metadata.
- Unit coverage exercises successful normalization, rate inconsistency,
  explicit ineligibility, and an eligibility flag contradicted by prefill
  collection metadata.
- The complete host-runtime gate passed 287 Rust tests, an optimized macOS
  build, the M1 Ultra Metal dispatch probe, 68 Python tests, every pinned
  differential fixture, and the Rust-writer/Python-reader C0 smoke test.
- The first optimized target-Mac 2K/128 development run reached Metal generation.
  A live process sample measured 22.3 GB current physical footprint and 66.9 GB
  peak footprint. The peak came from exhaustive native-prefill fixtures and
  host output materialization. The run ended before atomic JSON installation,
  so no throughput number is retained or claimed.

Decision:

- Keep the raw producer explicitly `paired_protocol_eligible: false` and make
  the adapter fail closed. The next change must split timing-only native prefill
  from the exact diagnostic collector rather than weakening C0 validation or
  publishing a memory-distorted rate.

Next:

- Give the prefill Metal ABI an execution-only path with no host boundary
  outputs, construct model descriptors without decoding fixtures, retain the
  existing correctness command unchanged, and repeat the exact 2K/128 run.

### 2026-08-27 — Exact native 2K handoff through production sparse position 4099

Objective:

- Transfer the retained exact batched-prefill state into the ordinary decoder
  representation without host readback, then prove the complete closed loop
  through the first default sparse ratio-4 step.

Evidence:

- Added a pinned-oracle capture that performs exact 2,048-token
  `ds4_session_sync`, then greedily evaluates positions 2048--4099. Two fresh
  processes produced the same 2,052 input-token transcript, immediate logits,
  final logits, and 51,659,152-byte retained prefill payload bit-for-bit.
- The compact fixture retains only the transcript and the two 129,280-logit
  frontiers. The full retained payload remains local diagnostic evidence and
  is not imported into the repository.
- Added one GPU-only adoption ABI. It copies all 43 raw-ring tails, layers
  2--42 attention compressed/recurrent state, and all even-layer indexer
  compressed/recurrent state in 229 blits under one command buffer and one
  host wait.
- The optimized M1 Ultra run matched the native prefill token 15342, every one
  of the 2,052 greedy selections, all position-2048 logits, and all
  position-4099 logits. At position 4099 all 21 even layers observed newly
  committed compressed row 1,025 and used the production two-block top-512
  sparse schedule. Input token 312 selected token 2538.
- The run reported 21,803.054/21,712.993 ms transformer wall/GPU time,
  262.624/2.705 ms output-head wall/GPU time, 78.904/0.207 ms handoff wall/GPU
  time, and 16.563 synchronized decode positions/s. These are correctness
  diagnostics with per-position waits and full frontier checks, not an engine
  throughput claim.

Decision:

- Native batched prefill, persistent-state ownership, exact post-prompt decode,
  and the first production sparse frontier are now one continuous C0 path.
  Captured initial state is no longer used by this control.

Next:

- Build a timing-specific 128-token engine-measurement producer from this
  exact state path, validate its artifact, and connect it to the paired runner.

### 2026-08-27 — Exact complete-model native 2K prefill through logits

Objective:

- Consume the retained final layer-42 HC row directly from the native batched
  prefill context and complete the exact output head without weakening any
  transformer-only regression boundary.

Evidence:

- Reused the independently repeated DwarfStar 2K batched-prefill frontier
  fixture. Its 129,280-logit payload has SHA-256
  `7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf`
  and selects token 15342 by lowest-ID argmax.
- Extended the proven decode output-head implementation to select either its
  retained one-row layer-42 buffer or row 2047 of the retained 2K prefill
  layer-42 HC buffer. Decode ownership and all prior transformer controls
  remain unchanged.
- The exact plain HC RMSNorm, F16 four-way HC projection, HC weighting, fused
  HC collapse/learned RMSNorm, and 129,280-row Q8_0 vocabulary projection add
  five dispatches and five pointer-identical model mappings. The complete
  native model schedule is now 2,377 dispatches and 1,221 no-copy mappings.
- The optimized focused M1 run matched every transformer boundary and every
  output logit bit-for-bit, selected token 15342, and reported 21478.015 ms
  wall / 21389.015 ms GPU for the terminal transformer schedule plus
  270.931 ms wall / 2.697 ms GPU for the correctness-readback output head.
  These intervals are correctness diagnostics, not throughput claims.
- The repeated target-model gate again matched all 129,280 logits and selected
  token 15342. It reported 32159.143 ms wall / 31397.883 ms GPU for the
  transformer schedule and 69.829 ms wall / 2.700 ms GPU for the output head.
  All 284 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  strict target-model inspection, and every native runtime control passed.
- The retained sequential-replay diagnostic remained exact against its decode
  oracle at 21.708 tokens/s over 94341.601 ms. Its intentional divergence from
  batched-prefill arithmetic still makes it ineligible for paired throughput.

Decision:

- Native batched model prefill is now exact and complete through full logits
  and deterministic greedy selection at the canonical 2K prompt boundary.
  Sparse post-prompt integration and an eligible throughput path remain
  outside this claim.

Next:

- Connect the retained native 2K state to closed-loop decoding and integrate
  the production-default sparse ratio-4 path before emitting paired engine
  measurements.

### 2026-08-27 — Exact complete layer-42 full-2K transformer prefill

Objective:

- Carry layer 41's retained final HC through the final transformer layer's
  complete native prefill path and validate the even-layer paired ratio-4
  attention/indexer compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-42 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and recurrent states, dense mixed attention, FFN, and
  both additive HC updates, chained from the layer-41 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-42 mappings and
  70 dispatches, taking the terminal schedule to 2,372 dispatches and
  1,216/1,216 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-42 attention/HC checksums bit-for-bit, reporting 20845.548 ms
  wall / 20741.050 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- The complete target-model regression repeated the 2K layers 0–42 schedule
  at 35813.130 ms wall / 35048.038 ms GPU with all 2,372 terminal dispatches
  and 1,216/1,216 no-copy mappings intact. The separate sequential frontier
  remained decode-replay C0 exact at 21.283 tokens/s over 96226.723 ms; its
  captured-state schedule is explicitly ineligible for a throughput claim.
- All 284 Rust tests and all 62 Python tests passed. The complete pinned
  differential corpus, all four new fixture bundles, strict target-model
  inspection, retained sparse controls, decoder/logit controls, and runtime
  benchmarks passed the one-command Mac Studio gate.

Decision:

- The exact native full-2K transformer prefill frontier is now complete
  through all layers 0–42. Output-head integration, complete-model native
  batched prefill, output logits, and a throughput-producing path remain
  outside this claim.

Next:

- Integrate the exact output head with the retained layer-42 final HC while
  preserving every transformer boundary as a regression control.

### 2026-08-27 — Exact complete layer-41 full-2K prefill

Objective:

- Carry layer 40's retained final HC through layer 41's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-41 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 compressor and recurrent state, dense mixed attention, FFN, and
  both additive HC updates, chained from the layer-40 complete fixture.
  Together they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-41 mappings and
  47 dispatches, taking the terminal schedule to 2,302 dispatches and
  1,184/1,184 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-41 attention/HC checksums bit-for-bit, reporting 31283.057 ms
  wall / 30976.413 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- The complete target-model gate repeated the boundary at 32546.263 ms wall /
  31779.314 ms GPU with the same exact dispatch and mapping counts.
- All 279 Rust tests and all 62 Python tests passed. All four new fixture
  bundles and the complete pinned fixture corpus passed independent manifest,
  size, and SHA-256 verification.
- The unchanged 2K sequential diagnostic reproduced the decode-replay logits
  exactly over 94268.320 ms at 21.725 tokens/s. This remains correctness
  evidence rather than a throughput claim for the native batched path.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 41.
  Layer 42, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 42 while preserving all
  retained layers-0–41 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-40 full-2K prefill

Objective:

- Carry layer 39's retained final HC through layer 40's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-40 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and their recurrent states, dense mixed attention, FFN,
  and both additive HC updates, chained from the layer-39 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-40 mappings and
  70 dispatches, taking the terminal schedule to 2,255 dispatches and
  1,156/1,156 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-40 attention/HC checksums bit-for-bit, reporting 17261.744 ms
  wall / 17169.696 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- All 275 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-40
  boundary at 25921.658 ms wall / 25162.075 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logit controls, and both benchmark smokes. Its 2K sequential
  diagnostic reproduced the final decode-replay logits exactly over
  96718.190 ms at 21.175 tokens/s. These are correctness diagnostics, not
  throughput claims.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 40.
  Layer 41, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 41 while preserving all
  retained layers-0–40 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-39 full-2K prefill

Objective:

- Carry layer 38's retained final HC through layer 39's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-39 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-38 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-39 mappings and
  47 dispatches, taking the terminal schedule to 2,185 dispatches and
  1,124/1,124 pointer matches.
- The first focused M1 preflight caught two mechanically shifted quantized
  row-size literals before execution. After restoring the correct 1,056-byte
  row size, the optimized correctness run matched every retained tensor and
  the full layer-39 attention/HC checksums bit-for-bit, reporting 15557.383 ms
  wall / 15471.540 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- All 270 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-39
  boundary at 24440.831 ms wall / 23671.163 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 96062.015 ms at 21.320 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 39.
  Layer 40, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 40 while preserving all
  retained layers-0–39 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-38 full-2K prefill

Objective:

- Carry layer 37's retained final HC through layer 38's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-38 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and their recurrent states, dense mixed attention, FFN,
  and both additive HC updates, chained from the layer-37 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-38 mappings and
  70 dispatches, taking the terminal schedule to 2,138 dispatches and
  1,096/1,096 pointer matches.
- The first focused M1 run localized four stale layer-36 compressor projection
  indices in the cloned layer-38 schedule. After correcting them, the
  optimized correctness run matched every retained tensor and the full
  layer-38 attention/HC checksums bit-for-bit, reporting 20538.412 ms wall /
  19774.631 ms GPU. This includes exhaustive correctness readback and is not a
  throughput claim.
- All 266 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-38
  boundary at 21674.890 ms wall / 21576.655 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 97252.434 ms at 21.059 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 38.
  Layer 39, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 39 while preserving all
  retained layers-0–38 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-37 full-2K prefill

Objective:

- Carry layer 36's retained final HC through layer 37's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-37 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-36 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-37 mappings and
  47 dispatches, taking the terminal schedule to 2,068 dispatches and
  1,064/1,064 pointer matches.
- All 261 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-37 attention/HC checksums bit-for-bit, reporting
  15705.615 ms wall / 14994.391 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-37
  boundary at 18044.576 ms wall / 17293.729 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 92544.759 ms at 22.130 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 37.
  Layer 38, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 38 while preserving all
  retained layers-0–37 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-36 full-2K prefill

Objective:

- Carry layer 35's retained final HC through layer 36's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-36 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved
  `b030961` oracle executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-35 complete fixture. Together
  they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-36 mappings and
  70 dispatches, taking the terminal schedule to 2,021 dispatches and
  1,036/1,036 pointer matches.
- All 257 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-36 attention/HC checksums bit-for-bit, reporting
  24203.696 ms wall / 24099.362 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-36
  boundary at 27551.388 ms wall / 26790.433 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 97424.253 ms at 21.021 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 36.
  Layer 37, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 37 while preserving all
  retained layers-0–36 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-35 full-2K prefill

Objective:

- Carry layer 34's retained final HC through layer 35's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-35 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved
  `b030961` oracle executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention-compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-34 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-35 mappings and
  47 dispatches, taking the terminal schedule to 1,951 dispatches and
  1,004/1,004 pointer matches.
- All 252 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-35 attention/HC checksums bit-for-bit, reporting
  31809.591 ms wall / 31050.835 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-35
  boundary at 25011.323 ms wall / 24213.220 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 94789.051 ms at 21.606 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 35.
  Layer 36, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 36 while preserving all
  retained layers-0–35 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-34 full-2K prefill

Objective:

- Carry layer 33's retained final HC through layer 34's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-34 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-33 complete fixture. Together
  they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-34 mappings and
  70 dispatches, taking the terminal schedule to 1,904 dispatches and 976/976
  pointer matches.
- All 248 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-34 attention/HC checksums bit-for-bit, reporting
  14317.612 ms wall / 13546.455 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-34
  boundary at 19871.581 ms wall / 19730.470 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 95319.280 ms at 21.486 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 34.
  Layer 35, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 35 while preserving all
  retained layers-0–34 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-33 full-2K prefill

Objective:

- Carry layer 32's retained final HC through layer 33's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-33 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 attention-compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-32 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-33 mappings and
  47 dispatches, taking the terminal schedule to 1,834 dispatches and 944/944
  pointer matches.
- All 243 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-33 attention/HC checksums bit-for-bit, reporting
  13126.804 ms wall / 12632.873 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-33
  boundary at 15919.880 ms wall / 15796.966 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 93650.728 ms at 21.868 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 33.
  Layer 34, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 34 while preserving all
  retained layers-0–33 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-32 full-2K prefill

Objective:

- Carry layer 31's retained final HC through layer 32's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-32 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-31 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-32 mappings and
  70 dispatches, taking the terminal schedule to 1,787 dispatches and 916/916
  pointer matches.
- All 239 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-32 attention/HC checksums bit-for-bit, reporting
  14678.233 ms wall / 14571.577 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-32
  boundary at 13283.525 ms wall / 13179.070 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 32.
  Layer 33, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 33 while preserving all
  retained layers-0–32 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-31 full-2K prefill

Objective:

- Carry layer 30's retained final HC through layer 31's complete native
  prefill path and validate its odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-31 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-30 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-31 mappings and
  47 dispatches, taking the terminal schedule to 1,717 dispatches and 884/884
  pointer matches.
- All 234 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-31 attention/HC checksums bit-for-bit, reporting
  16492.572 ms wall / 15726.058 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-31
  boundary at 9087.549 ms wall / 8984.633 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 31.
  Layer 32, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 32 while preserving all
  retained layers-0–31 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-30 full-2K prefill

Objective:

- Carry layer 29's retained final HC through layer 30's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-30 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-29 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-30 mappings and
  70 dispatches, taking the terminal schedule to 1,670 dispatches and 856/856
  pointer matches.
- All 230 Rust tests passed. All four new fixture bundles passed independent
  manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-30 attention/HC checksums bit-for-bit, reporting
  12541.042 ms wall / 12450.093 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-30
  boundary at 8843.842 ms wall / 8740.028 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 30.
  Layer 31, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 31 while preserving all
  retained layers-0–30 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-29 full-2K prefill

Objective:

- Carry layer 28's retained final HC through layer 29's complete native
  prefill path and validate its odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-29 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-28 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-29 mappings and
  47 dispatches, taking the terminal schedule to 1,600 dispatches and 824/824
  pointer matches.
- All 225 Rust tests passed. All four new fixture bundles passed independent
  manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-29 attention/HC checksums bit-for-bit, reporting
  13595.215 ms wall / 13486.902 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-29
  boundary at 8502.094 ms wall / 8380.564 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 29.
  Layer 30, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 30 while preserving all
  retained layers-0–29 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-28 full-2K prefill

Objective:

- Carry layer 27's retained final HC through layer 28's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-28 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-27 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-28 mappings and
  70 dispatches, taking the terminal schedule to 1,553 dispatches and 796/796
  pointer matches.
- Kept the layer-28 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 221 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-28 attention/HC checksums bit-for-bit, reporting
  23550.771 ms wall / 22839.249 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-28
  boundary at 14864.326 ms wall / 14745.832 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 28.
  Layer 29, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 29 while preserving all
  retained layers-0–28 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-27 full-2K prefill

Objective:

- Carry layer 26's retained final HC through layer 27's complete native
  prefill path and validate its odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-27 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-26 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-27 mappings and
  47 dispatches, taking the terminal schedule to 1,483 dispatches and 764/764
  pointer matches.
- Kept the layer-27 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 216 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-27 attention/HC checksums bit-for-bit, reporting
  20828.778 ms wall / 20099.925 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-27
  boundary at 9444.319 ms wall / 9320.928 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 27.
  Layer 28, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 28 while preserving all
  retained layers-0–27 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-26 full-2K prefill

Objective:

- Carry layer 25's retained final HC through layer 26's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-26 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-25 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-26 mappings and
  70 dispatches, taking the terminal schedule to 1,436 dispatches and 736/736
  pointer matches.
- Kept the layer-26 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 212 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-26 attention/HC checksums bit-for-bit, reporting
  18893.506 ms wall / 18128.441 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-model gate independently repeated the exact layer-26
  boundary at 6421.745 ms wall / 5754.788 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, exact 43-layer decoder/logits,
  the 2K sequential diagnostic, and both benchmark smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 26.
  Layer 27, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 27 while preserving all
  retained layers-0–26 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-25 full-2K prefill

Objective:

- Carry layer 24's retained final HC through layer 25's complete native
  prefill path and validate the odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-25 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-24 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-25 mappings and
  47 dispatches, taking the terminal schedule to 1,366 dispatches and 704/704
  pointer matches.
- Crossing layer 25 initially pushed embedded Rust fixture metadata beyond
  Mach-O's 4-GiB section limit. Layer-25 oracle payloads now load from their
  pinned fixture files at probe startup, preserving exact validation while
  removing that binary-growth failure.
- All 207 optimized Rust tests passed. The M1 Ultra focused correctness run
  matched every retained tensor and full attention/HC checksum bit-for-bit,
  reporting 14256.520 ms wall / 13414.574 ms GPU. This includes exhaustive
  correctness readback and is not a throughput claim.
- The complete target-model gate independently repeated the exact layer-25
  boundary at 12407.395 ms wall / 12278.251 ms GPU, then passed all 62 Python
  tests, the complete fixture corpus, the 2K sequential diagnostic, retained
  sparse controls, exact position-8195 decoder/logits, and both benchmark
  smokes. The 2K diagnostic was resumed after an operator interruption and
  completed in 126413.791 ms without rerunning the already-passed earlier
  stages.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 25.
  Layer 26, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 26 while preserving all
  retained layers-0–25 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-24 full-2K prefill

Objective:

- Carry layer 23's retained final HC through layer 24's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-24 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned and chained
  from the layer-23 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-24 model
  mappings and 70 dispatches, taking the complete terminal schedule to 1,319
  dispatches and 676/676 pointer matches.
- Kept the pinned 32-MiB full attention oracle runtime-loaded so the growing
  fixture corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- The focused gate initially exposed a mechanical fixture-name swap: the
  correct retained layer-22 ingress was compared against layer 24's oracle.
  Correcting the two constant names restored the retained regression without
  changing GPU execution.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 8274.750 ms
  wall and 8160.218 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 7757.827 ms wall /
  7637.127 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 203 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  100.1-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 24.
  Layer 25, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 25 while preserving all
  retained layers-0–24 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-23 full-2K prefill

Objective:

- Carry layer 22's retained final HC through layer 23's complete native
  prefill path and validate its ratio-128 compressor against DwarfStar.

Evidence:

- Captured 28 layer-23 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention compressor state, dense mixed attention, FFN, and both additive
  HC updates. Their complete captures are SHA-256-pinned and chained from the
  layer-22 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-23 model
  mappings and 47 dispatches, taking the complete terminal schedule to 1,249
  dispatches and 644/644 pointer matches.
- Kept the pinned 32-MiB full attention oracle runtime-loaded so the growing
  fixture corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 11244.205 ms
  wall and 10918.043 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 7274.859 ms wall /
  7166.631 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 198 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  95.3-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 23.
  Layer 24, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 24 while preserving all
  retained layers-0–23 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-22 full-2K prefill

Objective:

- Carry layer 21's retained final HC through layer 22's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-22 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-22 model
  mappings and 70 dispatches, taking the complete terminal schedule to 1,202
  dispatches and 616/616 pointer matches.
- The added full attention oracle crossed Mach-O's 4-GiB Rust metadata-section
  limit when embedded. Loading that pinned 32-MiB tensor from its versioned
  fixture at runtime keeps the exact bitwise comparison while restoring clean
  test builds.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 15146.208 ms
  wall and 15048.284 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 15966.815 ms wall /
  15175.116 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 194 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  98.7-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 22.
  Layer 23, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 23 while preserving all
  retained layers-0–22 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-21 full-2K prefill

Objective:

- Carry layer 20's retained final HC through layer 21's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-21 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-21 model mappings
  and 47 dispatches, taking the complete terminal schedule to 1,132 dispatches
  and 584/584 pointer matches.
- The first target-Mac run exposed four layer-21 weight references that had
  retained layer-19 mapping indices during mechanical rebasing. Correcting
  those indices made every retained tensor and full attention/HC checksum match
  bit-for-bit in two independent optimized M1 Ultra replays.
- The focused replay reported 6453.507 ms wall / 6312.943 ms GPU and the second
  replay reported 5785.641 ms wall / 5695.444 ms GPU. These are correctness
  timings with exhaustive readback, not throughput claims.
- The complete target-model gate repeated the boundary at 6325.002 ms wall /
  6208.810 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 189 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  normal 103.6-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 21.
  Layer 22, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 22 while preserving all
  retained layers-0–21 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-20 full-2K prefill

Objective:

- Carry layer 19's retained final HC through layer 20's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-20 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-20 model mappings
  and 70 dispatches, taking the complete terminal schedule to 1,085 dispatches
  and 556/556 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 17933.832 ms wall
  and 17082.936 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- Target-Mac validation passed optimized Objective-C/Metal compilation, all
  185 Rust tests, 62 Python tests, the complete pinned fixture corpus, every
  retained decoder/sparse control, and both benchmark smoke controls. It
  repeated the layer-20 boundary at 6004.181 ms wall and 5883.395 ms GPU.
- The comprehensive wrapper's unchanged 2K sequential frontier was stopped
  after more than 40 minutes of sustained compute under concurrent OBS load;
  its normal prior dedicated run was about 102 seconds. No result or timing
  from that interrupted control is claimed, and the wrapper did not reach its
  final success footer. This does not affect the two independent exact
  layer-20 replays or the directly rerun retained controls above.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 20.
  Layer 21, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 21 while preserving all
  retained layers-0–20 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-19 full-2K prefill

Objective:

- Carry layer 18's retained final HC through layer 19's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-19 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-19 model mappings
  and 47 dispatches, taking the complete terminal schedule to 1,015 dispatches
  and 524/524 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 5640.974 ms wall
  and 5542.788 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 180 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-19 boundary at
  5632.596 ms wall and 5524.921 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 102152.030 ms at 20.049 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 19.
  Layer 20, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 20 while preserving all
  retained layers-0–19 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-18 full-2K prefill

Objective:

- Carry layer 17's retained final HC through layer 18's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-18 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-18 model mappings
  and 70 dispatches, taking the complete terminal schedule to 968 dispatches
  and 496/496 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 8907.699 ms wall
  and 8044.939 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 176 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-18 boundary at
  5410.038 ms wall and 5298.708 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 100706.304 ms at 20.336 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 18.
  Layer 19, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 19 while preserving all
  retained layers-0–18 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-17 full-2K prefill

Objective:

- Carry layer 16's retained final HC through layer 17's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-17 tensors in ten accepted fresh DwarfStar processes over
  the canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical. One incomplete FFN capture was rejected and replaced in full.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-17 model mappings
  and 47 dispatches, taking the complete terminal schedule to 898 dispatches
  and 464/464 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 13618.961 ms wall
  and 12893.935 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 171 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-17 boundary at
  5050.306 ms wall and 4938.790 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 95270.878 ms at 21.497 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 17.
  Layer 18, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 18 while preserving all
  retained layers-0–17 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-16 full-2K prefill

Objective:

- Carry layer 15's retained final HC through layer 16's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-16 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-16 model mappings
  and 70 dispatches, taking the complete terminal schedule to 851 dispatches
  and 436/436 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 3760.783 ms wall
  and 3646.932 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 167 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-16 boundary at
  4752.235 ms wall and 4643.339 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 97842.871 ms at 20.932 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 16.
  Layer 17, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 17 while preserving all
  retained layers-0–16 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-15 full-2K prefill

Objective:

- Carry layer 14's retained final HC through layer 15's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-15 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-15 model mappings
  and 47 dispatches, taking the complete terminal schedule to 781 dispatches
  and 404/404 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 4477.729 ms wall
  and 4303.657 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 162 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-15 boundary at
  4504.211 ms wall and 4307.917 ms GPU.
- The first sequential-control attempt was terminated after 40 minutes by an
  app-session interruption without producing an artifact. An isolated recovery
  then reproduced the decode-replay logits exactly over 529388.007 ms at a
  diagnostic 3.869 tokens/s. This remains correctness evidence rather than a
  performance measurement; the incomplete batched prefill still differs from
  the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 15.
  Layer 16, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 16 while preserving all
  retained layers-0–15 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-14 full-2K prefill

Objective:

- Carry layer 13's retained final HC through layer 14's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-14 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-14 model mappings
  and 70 dispatches, taking the complete terminal schedule to 734 dispatches
  and 376/376 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 6931.003 ms wall
  and 6724.343 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac gate passed optimized Objective-C/Metal compilation,
  all 158 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  every retained decoder/sparse regression, and both benchmark smoke controls.
  It repeated the layer-14 boundary at 24431.809 ms wall and 22369.166 ms GPU
  under contention. Its 2K sequential control reproduced the decode-replay
  logits exactly over 5881468.976 ms at a diagnostic 0.348 tokens/s. These are
  correctness results, not performance measurements; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 14.
  Layer 15, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 15 while preserving all
  retained layers-0–14 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-13 full-2K prefill

Objective:

- Carry layer 12's retained final HC through layer 13's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-13 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-13 model mappings
  and 47 dispatches, taking the complete terminal schedule to 664 dispatches
  and 344/344 pointer matches.
- The optimized M1 Ultra correctness run matched all retained tensors and the
  full attention/HC checksums bit-for-bit. Its focused gate reported 19474.658
  ms wall and 17616.812 ms GPU under transient contention; the complete gate
  repeated the same boundary at 3869.312 ms wall and 3657.041 ms GPU. Both are
  correctness timings, not throughput claims.
- The complete target-Mac gate passed optimized Objective-C/Metal compilation,
  all 153 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  every retained decoder/sparse regression, and both benchmark smoke controls.
  Its contended 2K sequential control reproduced the decode-replay logits
  exactly over 7146464.377 ms at a diagnostic 0.287 tokens/s; that run is
  correctness evidence, not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 13.
  Layer 14, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 14 while preserving all
  retained layers-0–13 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-12 full-2K prefill

Objective:

- Carry layer 11's retained final HC through layer 12's complete native
  prefill path and validate the even-layer paired ratio-4 compressors
  independently against DwarfStar.

Evidence:

- Captured 31 layer-12 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-12 model mappings
  and 70 dispatches, taking the complete terminal schedule to 617 dispatches
  and 316/316 pointer matches.
- The optimized M1 Ultra correctness run matched all retained tensors and the
  full attention/HC checksums bit-for-bit. Its focused gate reported 3534.237
  ms wall and 3438.780 ms GPU; this includes exhaustive correctness readback
  and is not a throughput claim.
- The complete target-Mac gate repeated the same boundary at 3511.858 ms wall
  and 3422.943 ms GPU, then passed all 149 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. Its contended 2K sequential control still
  reproduced the decode-replay logits exactly at a diagnostic 0.328 tokens/s;
  that run is correctness evidence, not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 12.
  Layer 13, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 13 while preserving all
  retained layers-0–12 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-11 full-2K prefill

Objective:

- Carry layer 10's retained final HC through layer 11's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-11 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates.
- Extended the persistent Metal context with 28 no-copy layer-11 model mappings
  and 47 dispatches, taking the complete terminal schedule to 547 dispatches
  and 284/284 pointer matches.
- The optimized M1 Ultra correctness run matched all 28 retained tensors and
  the full attention/HC checksums bit-for-bit. Its focused gate reported
  3193.051 ms wall and 3098.131 ms GPU; this includes exhaustive correctness
  readback and is not a throughput claim.
- The final target-Mac gate passed optimized Objective-C/Metal compilation,
  144 Rust tests, 62 Python tests, the complete pinned fixture corpus, and every
  retained decoder/sparse regression. Its contended 2K sequential control still
  reproduced the decode-replay logits exactly; that run is correctness evidence,
  not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 11.
  Layer 12, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 12 while preserving the
  complete layers-0–11 command as a regression control.

### 2026-08-24 — Exact complete layer-10 full-2K prefill

Objective:

- Carry layer 9's retained final HC through layer 10's complete native prefill
  path and validate the even-layer paired compressors independently against
  DwarfStar.

Evidence:

- Captured 31 layer-10 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired ratio-4
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates.
- Extended the persistent Metal context with 32 no-copy layer-10 model mappings
  and 70 dispatches, taking the complete terminal schedule to 500 dispatches
  and 256/256 pointer matches.
- The optimized M1 Ultra correctness run matched all 31 retained tensors and
  the full attention/HC checksums bit-for-bit. Its final gate reported 2871.412
  ms wall and 2792.469 ms GPU; this includes exhaustive correctness readback
  and is not a throughput claim.
- The complete host-runtime gate passed after the focused hardware run:
  formatting, Objective-C/Metal compilation, optimized build, all 140 Rust
  tests, 62 Python tests, every pinned differential fixture, and the
  cross-language C0 artifact contract.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 10.
  Layer 11, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 11, then reassess whether
  to continue layer-by-layer or connect the retained prefill state more directly
  to the existing decoder/output path.

### 2026-08-24 — Exact complete layer-9 full-2K prefill

Objective:

- Carry layer 8's retained final HC through layer 9's complete native prefill
  path and validate the new boundary independently against DwarfStar.

Evidence:

- Captured 28 layer-9 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  compressor state, dense mixed attention, FFN, and both additive HC updates.
- Extended the persistent Metal context with 28 no-copy layer-9 model mappings
  and 47 dispatches, taking the complete terminal schedule to 430 dispatches
  and 224/224 pointer matches.
- The optimized M1 Ultra correctness run matched all 28 retained tensors and
  the full attention/HC checksums bit-for-bit. It reported 2535.036 ms wall and
  2462.104 ms GPU; this includes exhaustive correctness readback and is not a
  throughput claim.
- Rust formatting, Objective-C/Metal compilation, and 131 unit tests passed
  before the focused hardware run.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 9.
  Layer 10, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 10, then reassess whether
  to continue layer-by-layer or connect the retained prefill state more directly
  to the existing decoder/output path.

### 2026-08-23 — Retained sparse top-k generalized through repeated merges

Objective:

- Remove the first-boundary guard and prove the production retained scheduler
  at the first row count that requires more than one top-k merge pass.

Implementation:

- Traced DwarfStar's complete sort schedule: choose the largest power-of-two
  sort width supported by the pipeline, retain up to 512 indices per initial
  block, derive the compact work width from the final partial block, then
  ping-pong pairwise merge passes until one exact top-512 remains.
- Added `retained-sparse-layer2-pos8195-v1`, a strict 38-operation, 47-tensor,
  6,567,092-byte fixture from two fresh 8K DwarfStar processes. It seeds a
  wrapped 127-row raw-ring window, 2,048 rows in both compressed caches, and all
  four recurrent states immediately before layer 2 position 8195.
- Replaced the fixed 513-index scratch and single merge with capacity-stable
  ping-pong storage plus visible-row schedule derivation. At row 2,049 this
  produces three 1,024-thread initial blocks, a 1,025-index active work width,
  and two merge dispatches. The original 1,025-row schedule remains one merge.
- Added `retained-sparse-multimerge-probe`, its stable JSON contract, CLI and
  runtime-gate integration. The command continues to deny preceding-layer,
  token-dependent FFN, complete-decoder, logits, and throughput claims.
- Removed all temporary DwarfStar capture hooks immediately after capture;
  `ds4.c` is unchanged. Two diagnostic-only oversized clamped expert scratch
  dumps had non-semantic fresh-process differences and were excluded; every
  seed, sparse tensor, weighted expert result, routed output, and final HC was
  byte-identical.

Validation:

- Strict fixture verification accepted 38 operations, 47 tensors, and
  6,567,092 bytes. Rust host suite: 129 tests passed. Optimized
  Rust/Objective-C/Metal compilation passed on the M1 Ultra.
- Focused runs kept both retained regressions C0 exact. Row 1,025 still matched
  16 tensors over 54 dispatches with 35/35 mappings. Row 2,049 matched 16
  tensors over 55 dispatches with 35/35 mappings and reported 61.399 ms wall /
  32.045250 ms GPU with correctness setup/readback in scope.
- The complete target-Mac gate passed 129 Rust tests, 61 Python tests, strict
  fixture validation, optimized Rust/Objective-C/Metal compilation, and every
  model-backed control. It repeated row 1,025 C0 exact at 51.990 ms wall /
  28.167000 ms GPU and row 2,049 at 49.455 ms wall / 28.882250 ms GPU.

Next:

- Drive the generalized schedule from complete retained decoder progression,
  then use the eligible native prefill/decode loop for engine measurements.

### 2026-08-23 — Retained state crosses the first default sparse row

Objective:

- Prove that the production retained layer executor, rather than only the
  isolated sparse diagnostic, reaches DwarfStar's first default sparse row with
  the correct cache, recurrent-state, and same-step update ordering.

Implementation:

- Captured two fresh DwarfStar layer-2 position-4099 controls and required exact
  agreement for the incoming HC, four pre-update compressor states, and all
  sparse-boundary outputs. Temporary capture hooks were removed immediately;
  `ds4.c` remains unchanged.
- Added `retained-sparse-layer2-pos4099-v1`, a strict 38-operation, 47-tensor,
  3,941,556-byte layer-segment fixture, plus a deterministic importer. The two
  score-state payloads preserve their exact non-finite bit patterns as integer
  tensors and are reinterpreted only at the runtime boundary.
- Added a diagnostic seed API that populates the exact persistent Metal keys
  used by the general retained executor: the incoming layer-2 HC, 127 raw ring
  rows, 1,024 attention/indexer compressed rows, and both recurrent states.
  Queue-ordered empty predecessors preserve the scheduler's declared layer-2
  chain ownership without claiming that layers 0 and 1 were executed.
- Added `retained-sparse-boundary-probe`. It executes the normal retained
  layer-2 position-4099 schedule, commits compressed row 1,025, runs the
  two-block top-k merge and 12-split indexed attention, and verifies 16 tensors
  by FP32 or integer bit pattern. The JSON explicitly denies complete-layer,
  complete-decoder, output-logit, and throughput claims because the captured HC
  bypasses preceding layers and the placeholder token is not an FFN oracle.

Validation:

- Strict fixture verification accepted 38 operations, 47 tensors, and
  3,941,556 bytes. The two fresh source captures matched for every new seed
  tensor.
- Rust host suite: 126 tests passed. Optimized Rust/Objective-C/Metal
  compilation and the focused M1 Ultra model run passed.
- The focused retained-state run matched all 16 tensors, preserved 35/35 model
  mmap pointers, and submitted 54 dispatches. It reported 53.413 ms wall /
  29.894125 ms GPU with seed, synchronization, and exhaustive readback in
  scope; this is correctness evidence only.
- The complete target-Mac gate passed 126 Rust tests, 61 Python tests, strict
  fixture validation, optimized Rust/Objective-C/Metal compilation, and every
  model-backed control. Its retained-state replay was C0 exact at 58.511 ms
  wall / 30.369250 ms GPU with the same 54 dispatches and 35/35 mappings.

Next:

- Generalize the retained argsort/merge workspace and repeated merge schedule
  beyond 1,025 compressed rows, then use it in a complete decoder progression.

### 2026-08-23 — Production-default sparse switch captured and wired

Objective:

- Replace the diagnostic threshold override with evidence at DwarfStar's real
  first sparse row and connect that exact schedule to retained decoder state.

Implementation:

- Ran two fresh 4K DwarfStar processes without a sparse-threshold override and
  captured layer 2 at position 4099. Every sparse-relevant payload was
  byte-identical: 1,025 score values, exact top-512 indices, both compressed
  caches, the physical raw cache, indexed attention, and inverse RoPE output.
- Added `sparse-indexed-attention-pos4099-v1`, a 12-tensor, 3,339,012-byte
  differential fixture. The importer preserves the older position-2051
  override and selects the default-boundary profile explicitly.
- Imported DwarfStar's two-block argsort merge into the isolated schedule. The
  public probe now executes the 513-row control first and then the 1,025-row
  production-default fixture.
- Extended retained even-layer ABI ownership with the two sparse indexer model
  spans. At row 1,025 it commits the newly emitted attention/indexer rows before
  scoring, runs the exact indexer projection/RoPE/QAT/score/sort/merge and
  12-split indexed attention schedule, then continues through the established
  inverse-RoPE, output, and FFN path.
- Kept an explicit guard after row 1,025. Repeated merge passes and workspace
  sizing for larger contexts remain a separate correctness step.
- Temporary DwarfStar dump hooks were removed; `ds4.c` is clean. The two local
  manual evidence logs remain private and untracked.

Validation:

- Both new DwarfStar captures matched bit-for-bit at every retained sparse
  tensor. The strict fixture verifier accepted 10 operations, 12 tensors, and
  3,339,012 bytes.
- Rust host suite: 124 tests passed. Optimized Rust/Objective-C/Metal compilation
  passed on the M1 Ultra.
- Focused model run executed both sparse controls C0 exact. The reported default
  boundary used 11 dispatches, preserved 3/3 mmap pointers, and measured
  25.162 ms wall / 0.550000 ms GPU with correctness setup/readback in scope.
- The complete target-Mac gate passed 124 Rust tests, 61 Python tests, strict
  validation of both sparse fixtures and every retained model-backed control.
  Its default-boundary replay was C0 exact at 22.303 ms wall / 0.547833 ms GPU.
- No complete retained decoder, output-logit, or throughput claim was made for
  the newly wired branch in this checkpoint.

Next:

- Add a retained-state C0 control that actually reaches row 1,025, then
  generalize the merge schedule beyond the first boundary before producing an
  engine measurement.

### 2026-08-23 — Exact diagnostic top-512 sparse indexed attention

Objective:

- Validate DwarfStar's complete ratio-4 indexer selection and sparse indexed
  attention mechanism on the M1 Ultra before changing retained decoder state.

Correction:

- The earlier journal wording that sparse selection begins immediately after
  512 compressed rows was incomplete. Pinned source commit `b030961` uses a
  default sparse threshold of 1,024 and a fixed model top-k of 512; the default
  first sparse row count is 1,025. A diagnostic environment override may lower
  the threshold without changing the production default.

Implementation:

- Captured layer 2 at position 2051 twice in fresh DwarfStar processes with
  `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=512`. All 13 dump artifacts were
  byte-identical, including indexer Q/weights/scores, exact top-512 indices,
  indexed-attention output, and inverse-RoPE output.
- Added a strict importer and a 12-tensor, 2,026,244-byte differential fixture.
  Its manifest records the override, pinned default, physical-to-compact raw
  cache slice, and explicit denial of default-switch, complete-decode, logits,
  and throughput claims.
- Imported DwarfStar's direct score, exact argsort, 12-way indexed attention,
  and reduction kernels. Added one no-copy ABI/CLI command that also reuses the
  existing F16 matvec, RoPE, indexer-QAT, and F32-to-F16 conversion kernels.
- Aligned Rust Star's dense-path guard with the pinned 1,024-row default. The
  isolated sparse path is not yet connected to retained 43-layer decoder state.

Validation:

- Fixture integrity validation passed: 9 operations, 12 tensors, 2,026,244
  verified bytes.
- Focused optimized M1 Ultra run: all retained values C0 exact, 10 dispatches,
  3/3 mmap pointer identities, 18.864 ms wall / 0.497625 ms GPU. This includes
  correctness setup/readback and is not a throughput measurement.
- Full target-Mac gate passed: 124 Rust tests, 61 Python tests, all prior
  model-backed controls, cross-language artifact validation, and the new
  fixture/probe. The gate sparse run repeated C0 exact with 3/3 mmap pointers
  at 21.616 ms wall / 0.538250 ms GPU; this remains diagnostic only.

Next:

- Integrate the sparse schedule into retained decoder ownership at 1,025
  compressed rows, with the 513-row override remaining an independent kernel
  boundary; then continue native prefill through layer 9.

### 2026-08-22 — Exact complete layer-8 full-2K prefill

Objective:

- Carry layer 7's retained final HC directly through layer 8's complete native
  prefill schedule without a host activation handoff.

Implementation:

- Captured layer 8's ten Q/KV boundaries, paired ratio-4 attention/indexer
  compressor outputs and states, dense-attention diagnostics/output/HC post,
  and complete routed/shared FFN in independent fresh DwarfStar processes;
  every paired full capture was byte-identical.
- Added four differential fixtures covering 31 retained tensors and 53,134,848
  bytes, with full-capture SHA-256 identities retained in their manifests.
- Wrapped 32 additional GGUF spans without copies, extended the shared Metal
  schedule by 70 dispatches, retained all layer-8 state in the persistent
  context, and added exhaustive C0 comparisons plus stable JSON reporting.
- Implemented layer 8's paired ratio-4 compressor ownership for both attention
  and indexer state. At the exact 2K boundary all 512 compressed rows remain on
  the dense path; sparse indexer top-k is explicitly unclaimed.
- Updated the CLI and runtime documentation. The artifact closes at
  `layer8_ffn_hc_post` and explicitly denies sparse post-prompt attention,
  complete-model-prefill, output-logit, and throughput claims.

Validation:

- All four new differential fixtures validate independently.
- Rust unit suite: 122 passed. Python tooling suite: 61 passed.
- Focused optimized M1 Ultra run: every retained layer-8 boundary C0 exact,
  2241.471 ms wall / 2153.893625 ms GPU, 383 dispatches, and 196/196 no-copy
  model mappings.
- Full target-Mac gate: all host/runtime controls passed; the complete
  layers-0–8 command reported 2289.152 ms wall / 2161.306708 ms GPU with the
  same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims.

Next:

- Implement the fixed 512-row ratio-4 sparse-indexed decode boundary so
  generation can advance beyond the native 2K prompt, then continue the exact
  batched-prefill frontier through layer 9.

### 2026-08-22 — Exact complete layer-7 full-2K prefill

Objective:

- Carry layer 6's retained final HC directly through layer 7's complete native
  prefill schedule without a host activation handoff.

Implementation:

- Captured layer 7's ten Q/KV boundaries, ratio-128 compressor output/state,
  dense-attention diagnostics/output/HC post, and complete routed/shared FFN in
  independent fresh DwarfStar processes; every paired full capture was
  byte-identical.
- Added four differential fixtures covering 28 retained tensors and
  52,299,264 bytes, with full-capture SHA-256 identities retained in their
  manifests.
- Wrapped 28 additional GGUF spans without copies, extended the shared Metal
  schedule by 47 dispatches, retained all layer-7 state in the persistent
  context, and added exhaustive C0 comparisons and stable JSON reporting.
- Initialized layer 7's compressed-position buffer with the canonical
  ratio-128 positions; the first device run exposed this otherwise-silent
  ownership requirement at the compressed-KV boundary.
- Restored the standalone Q8 projection encoder lifecycle after the initial
  schedule-block relocation removed its original `endEncoding`; the projection
  regression control is C0 exact again.
- Updated the CLI and runtime documentation. The artifact closes at
  `layer7_ffn_hc_post` and explicitly denies layer-8, sparse post-prompt,
  complete-model-prefill, output-logit, and throughput claims.

Validation:

- All four new differential fixtures validate independently.
- Rust unit suite: 117 passed. Python tooling suite: 61 passed.
- Focused optimized M1 Ultra run: every retained layer-7 boundary C0 exact,
  1950.645 ms wall / 1864.381 ms GPU, 313 dispatches, and 164/164 no-copy model
  mappings.
- Full target-Mac gate: all host/runtime controls passed; the complete
  layers-0–7 command reported 1915.398 ms wall / 1819.682 ms GPU with the same
  schedule and mapping counts. These intervals include exhaustive correctness
  readback and are not throughput claims.

Next:

- Extend the exact terminal prefill frontier through layer 8, or prioritize the
  fixed 512-row ratio-4 sparse-indexed decode boundary if decode progression is
  the nearer benchmark gate.

### 2026-08-22 — Exact complete layer-6 FFN and final HC

Objective:

- Continue the retained layer-6 attention HC post through its routed/shared FFN
  and additive final HC update without a host activation handoff.

Implementation:

- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/scaled
  weights, weighted routed SwiGLU, routed output, shared-expert output, and
  `hc_ffn_post` in two fresh DwarfStar processes; all ten full repeated captures
  were byte-identical.
- Added `prefill-layer6-complete-2048-v1`, retaining exact final-tile payloads
  while SHA-256-pinning every full 2K tensor identity, including the complete
  128 MiB final HC state.
- Wrapped layer 6's twelve FFN model spans directly from the GGUF mmap and
  appended the proven 21-dispatch HC ingress, biased top-6 router,
  IQ2_XXS/Q2_K routed expert, Q8_0 shared expert, and additive HC schedule.
- Extended the C ABI, retained Metal ownership, exhaustive Rust comparisons,
  stable JSON, CLI, documentation, fixture verification, and target-Mac gate.
  The artifact now closes at `layer6_ffn_hc_post` and explicitly denies layer-7
  prefill, sparse post-prompt top-k, complete-model-prefill, and throughput
  claims.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 113 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1709.387 ms wall
  / 1620.434 ms GPU, 266 dispatches, and 136/136 no-copy model mappings.
- Full target-Mac gate: 113 Rust tests and 61 Python tests passed; every retained
  runtime control remained exact, and the complete layers-0–6 command reported
  1636.955 ms wall / 1548.723 ms GPU with 136/136 mappings. These intervals
  include exhaustive correctness readback and are not throughput claims.

Next:

- Carry the retained layer-6 final HC into layer 7 and extend the complete
  full-2K prefill frontier one layer deeper.

### 2026-08-22 — Exact layer-6 dense mixed attention HC post

Objective:

- Continue the retained layer-6 paired-compressor state through dense mixed
  attention and its additive HC post without a host activation handoff.

Implementation:

- Captured attn_out, hc_attn_post, kqv_out, kqv_back, and attn_low in two fresh
  DwarfStar processes; all five full repeated captures were byte-identical.
- Added prefill-layer6-attention-2048-v1, retaining the complete 32 MiB
  attention output, three row-0 diagnostics, and final 32-row HC tile while
  checksum-pinning the complete 128 MiB HC identity.
- Wrapped layer 6's attention sinks and two Q8 output projections directly from
  the GGUF mmap and appended the proven nine-dispatch dense mixed-attention
  schedule.
- Added explicit retained buffers for layer 5's KQV-back and attention-low
  diagnostics. Those values previously remained in shared scratch only; layer
  6 attention correctly reuses that scratch, so the snapshots preserve the
  already-published layer-5 evidence boundary without adding dispatches.
- Extended the C ABI, exhaustive Rust comparisons, stable JSON, CLI,
  documentation, fixture verification, and target-Mac gate. The artifact closes
  at layer6_attention_hc_post and explicitly denies layer-6 FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 112 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1545.043 ms wall
  / 1464.156 ms GPU, 245 dispatches, and 124/124 no-copy model mappings.
- Full target-Mac gate: 112 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1635.054 ms wall / 1454.353 ms GPU with 124/124
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's biased-top-6 routed/shared FFN and additive final
  HC update.

### 2026-08-22 — Exact layer-6 paired ratio-4 compressors

Objective:

- Continue the retained layer-6 Q/KV state through both native ratio-4
  attention/indexer compressors without a host activation handoff.

Implementation:

- Captured KVcompress, attn_state_kv, attn_state_score,
  indexer_KVcompress, indexer_state_kv, and indexer_state_score in two fresh
  DwarfStar processes; all six repeated captures were byte-identical.
- Added prefill-layer6-compressors-2048-v1, containing all 512 attention and
  indexer compressed rows plus the four final recurrent-state tensors.
- Wrapped the eight layer-6 compressor tensors directly from the GGUF mmap and
  appended the proven 30-dispatch paired ratio-4 schedule to the retained
  terminal Metal command.
- Extended the C ABI, exact Rust comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact closes at
  layer6_paired_compressors and explicitly denies layer-6 attention, FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid six-tensor, 1,392,640-byte differential fixture.
- Rust unit suite: 110 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1358.621 ms wall
  / 1266.291 ms GPU, 236 dispatches, and 121/121 no-copy model mappings.
- Full target-Mac gate: 110 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1365.469 ms wall / 1265.535 ms GPU with 121/121
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's dense mixed attention and additive attention HC
  post.

### 2026-08-22 — Exact layer-6 full-2K Q/KV state

Objective:

- Continue the retained layer-5 final HC state through layer 6's native Q/KV
  path without a host activation handoff.

Implementation:

- Captured `hc_attn_pre`, `attn_norm`, `q_lora`, `q_lora_norm`, `KVraw`,
  `KVnorm`, `Qraw`, `Qcur`, `KVrope`, and `KVcur` in four fresh DwarfStar
  processes. Both primary captures and both isolated `Qraw` captures were
  byte-identical.
- Added `prefill-layer6-qkv-2048-v1`, retaining exact final 32-row tiles while
  SHA-256-pinning the complete 2K identities.
- Wrapped layer 6's nine HC/QKV tensors directly from the GGUF mmap and appended
  the proven ten-dispatch Q/KV schedule to the retained Metal command.
- Extended the C ABI, Rust exact comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact now closes at
  `layer6_qkv_state` and explicitly denies layer-6 compressors, attention, FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid ten-tensor, 9,961,472-byte differential fixture.
- Rust unit suite: 109 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1338.542 ms wall
  / 1247.536 ms GPU, 206 dispatches, and 113/113 no-copy model mappings.
- Full target-Mac gate: 109 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1328.350 ms wall / 1241.698 ms GPU with 113/113
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's paired ratio-4 attention/indexer compressors.

### 2026-08-22 — Exact complete layer-5 full-2K prefill

Objective:

- Continue the retained layer-5 post-attention HC state through its complete
  biased-top-6 routed/shared FFN and additive final HC update.

Implementation:

- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/scaled
  weights, routed weighted-SwiGLU/output, shared-expert output, and
  `hc_ffn_post` twice in fresh DwarfStar processes. All ten complete tensors
  were byte-identical.
- Added `prefill-layer5-complete-2048-v1`, retaining the exact final 32-row
  comparison tiles while SHA-256-pinning all complete source tensors and the
  complete final HC identity.
- Wrapped layer 5's twelve FFN tensors directly from the GGUF mmap and appended
  the proven 21-dispatch biased-top-6 schedule to the retained Metal command.
- Extended the C ABI, exact Rust comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact now closes at
  `layer5_ffn_hc_post` and explicitly denies later-layer, sparse-attention,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 108 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1358.984 ms wall
  / 1270.350 ms GPU, 196 dispatches, and 104/104 no-copy model mappings.
- Full target-Mac gate: 108 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1363.509 ms wall / 1272.876 ms GPU with 104/104
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Begin layer 6's native Q/KV boundary from the retained layer-5 final HC state.

### 2026-08-22 — Exact layer-5 full-2K dense mixed attention

Objective:

- Continue the retained layer-5 Q/KV and ratio-128 compressor state through
  dense mixed attention and the additive attention HC post-state.

Implementation:

- Captured `kqv_out`, `kqv_back`, `attn_low`, `attn_out`, and `hc_attn_post`
  twice in fresh DwarfStar processes. All five complete tensors were byte-stable.
- Added `prefill-layer5-attention-2048-v1`, retaining the complete 32 MiB
  attention output, three row-zero diagnostics, and the final 32-row HC tile
  while SHA-256-pinning the complete 128 MiB HC identity.
- Wrapped layer 5's sinks and two Q8 output projections directly from the model
  mmap and extended the retained Metal command with the proven nine-dispatch
  ratio-128 dense-attention schedule.
- Snapshotted layer 4's shared attention diagnostics before layer 5 reuses the
  scratch buffers; this preserves prior evidence without adding a dispatch.
- Extended the Rust ABI, exhaustive C0 checks, stable JSON, CLI, documentation,
  and target-Mac gate label. The layer-5 FFN remains explicitly out of scope.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 107 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1242.696 ms wall
  / 1152.157 ms GPU, 175 dispatches, and 92/92 no-copy model mappings. This
  includes exhaustive correctness readback and is not a throughput claim.
- Full target-Mac gate: 107 Rust tests and 61 Python tests passed, every native
  regression control remained exact, and the extended command reported
  1253.568 ms wall / 1159.442 ms GPU with 92/92 no-copy mappings.

Next:

- Continue the retained post-attention HC state through the complete layer-5
  FFN and additive final HC update.

### 2026-08-22 — Exact layer-5 full-2K ratio-128 compressor

Objective:

- Continue the retained layer-5 Q/KV state through all 16 ratio-128 emissions
  and establish the exact final recurrent-state boundary before dense attention.

Implementation:

- Captured `KVcompress`, `attn_state_kv`, and `attn_state_score` twice in fresh
  DwarfStar processes. Both captures were byte-identical; the aligned 2K batch
  ends with canonical zero KV scratch and negative-infinity score scratch.
- Added `prefill-layer5-compressor-2048-v1`, pinning 557,056 bytes across the
  16x512 compressed output and two 128x512 final state tensors.
- Wrapped four additional layer-5 compressor tensors directly from the model
  mmap and extended the retained Metal command with two F16 projections plus
  the proven five-dispatch ratio-128 batch schedule.
- Extended the Rust ABI, exhaustive C0 comparisons, stable JSON boundary, CLI,
  documentation, and target-Mac gate label. Layer-5 attention and FFN remain
  explicitly out of scope.

Validation:

- Fixture verifier: valid three-tensor, 557,056-byte differential fixture.
- Rust unit suite: 105 passed.
- Focused optimized M1 Ultra run: all 16 emissions and both terminal states C0
  exact, 1076.840 ms wall / 991.235 ms GPU, 166 dispatches, and 89/89 no-copy
  model mappings. This includes exhaustive correctness readback and is not a
  throughput claim.
- Full target-Mac gate: 105 Rust tests and 61 Python tests passed, every native
  regression control remained exact, and the extended command reported
  1072.662 ms wall / 972.161 ms GPU with 89/89 no-copy mappings.

Next:

- Continue the retained layer-5 state through dense mixed attention, additive
  attention HC post-processing, and the complete FFN.

### 2026-08-22 — Exact layer-5 HC ingress and full-2K Q/KV state

Objective:

- Carry layer 4's retained final HC directly into layer 5 and establish an
  exact, GPU-resident Q/KV checkpoint for the next ratio-128 compressor step.

Implementation:

- Captured layer-5 HC attention ingress, learned norm, Q-Lora, Q-Lora norm,
  raw/current Q, and raw/normalized/RoPE/final KV across four fresh DwarfStar
  processes. Every repeated full tensor was byte-stable.
- Added `prefill-layer5-qkv-2048-v1`, retaining compact final 32-row tiles for
  ten boundaries while pinning each complete tensor identity by SHA-256.
- Wrapped nine additional layer-5 tensors directly from the model mmap and
  extended the terminal command with the proven ten-dispatch HC/QKV schedule.
  The full finalized Q and KV buffers remain in the persistent Metal context.
- Extended stable JSON to the `layer5_qkv_state` boundary with explicit layer-5
  correctness and scope claims.

Validation:

- Fixture verifier: valid ten-tensor, 9,961,472-byte differential fixture.
- Rust unit suite: 104 passed.
- Focused optimized M1 Ultra run: all ten layer-5 boundaries C0 exact,
  1085.161 ms wall / 994.834 ms GPU, 159 dispatches, and 85/85 no-copy model
  mappings.
- Full target-Mac gate: 104 Rust tests and 61 Python tests passed, every native
  control remained exact, and the extended command reported 1088.225 ms wall /
  991.314 ms GPU with 85/85 no-copy mappings.

Next:

- Continue the retained full layer-5 Q/KV state through all 16 ratio-128
  compressor emissions and its final recurrent state.

### 2026-08-22 — Complete exact layer-4 prefill

Objective:

- Continue the retained layer-4 post-attention HC state through the complete
  FFN and establish native full-2K ownership through layer 4.

Implementation:

- Captured ten FFN hooks twice in fresh DwarfStar processes: HC ingress,
  learned norm, router logits/probabilities, biased top-6 selections, scaled
  weights, fused routed activations/output, shared output, and final HC. All
  full-2K identities were byte-stable.
- Added `prefill-layer4-complete-2048-v1`, retaining exact final-tile payloads
  and pinning the complete 128 MiB final-HC identity by SHA-256 and checksum.
- Wrapped twelve additional layer-4 FFN tensors directly from the model mmap
  and added the proven 21-dispatch biased router, routed IQ2_XXS/Q2_K expert,
  shared Q8_0 expert, and additive HC schedule.
- Extended the stable JSON boundary to `layer4_ffn_hc_post`, with 149 terminal
  dispatches, 76/76 no-copy mappings, exact FFN flags, and a complete layer-4
  prefill claim.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 103 passed.
- Focused optimized M1 Ultra run: every retained FFN boundary and the complete
  final HC identity C0 exact, 1079.281 ms wall / 991.837 ms GPU.
- Full target-Mac gate: 103 Rust tests and 61 Python tests passed, every native
  control remained exact, and the complete layer-4 command reported 1601.151
  ms wall / 985.798 ms GPU with 76/76 no-copy mappings.

Next:

- Continue the retained final HC state into layer 5, beginning with its HC
  attention ingress and Q/KV state.

### 2026-08-22 — Exact layer-4 dense mixed attention and HC post

Objective:

- Continue the retained full-2K layer-4 Q/KV and paired-compressor state
  through dense mixed attention and the additive four-stream HC update without
  a host activation upload.

Implementation:

- Captured `kqv_out`, `kqv_back`, `attn_low`, `attn_out`, and `hc_attn_post`
  twice in fresh DwarfStar processes. Every capture was byte-stable.
- Added `prefill-layer4-attention-2048-v1`, retaining the complete 2048x4096
  attention output, final 32-row HC tile, and compact row-zero diagnostics.
- Wrapped layer 4's sinks and two attention-output weights directly from the
  model mmap. Added the F32/F16 staging, FlashAttention, inverse RoPE, grouped
  map, low/output projections, and HC expansion as nine terminal dispatches.
- Snapshotted layer 3's shared diagnostic buffers before their reuse by layer
  4. The first focused run exposed that lifetime overlap; the corrected run
  passed all retained layer-3 and layer-4 comparisons.
- Extended the stable JSON boundary to `layer4_attention_hc_post`, with 128
  terminal dispatches, 64/64 no-copy mappings, and pinned full-output checksums.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 102 passed.
- Focused optimized M1 Ultra run: complete attention output and HC post C0
  exact, 934.193 ms wall / 833.479 ms GPU.
- Full target-Mac gate: 102 Rust tests and 61 Python tests passed, every native
  control remained exact, and the terminal command reported 952.654 ms wall /
  851.603 ms GPU with 64/64 no-copy mappings.

Next:

- Complete the layer-4 FFN from the retained post-attention HC state.

### 2026-08-22 — Exact layer-4 paired ratio-4 compressors

Objective:

- Continue the retained full-2K layer-4 normalized activation through both
  even-layer ratio-4 compressors without a host activation upload.

Implementation:

- Captured `KVcompress`, `attn_state_kv`, `attn_state_score`,
  `indexer_KVcompress`, `indexer_state_kv`, and `indexer_state_score` twice in
  fresh DwarfStar processes. All six identities were byte-stable.
- Added `prefill-layer4-compressors-2048-v1`, containing 512 attention rows,
  512 indexer rows, and all four final recurrent-state tensors.
- Wrapped eight additional layer-4 compressor tensors directly from the model
  mmap and added four full-batch F16 projections plus exact ratio-4 replay
  pooling, learned norm, compressed RoPE, E4M3FN/indexer QAT, and final state
  refresh.
- Corrected the fused ratio-4 shift kernel to restore the consumed upper state
  bank to DwarfStar's canonical zero/negative-infinity scratch state. The first
  focused comparison localized the issue at the exact four-row bank boundary;
  the corrected run passed every C0 comparison.
- Extended the stable JSON boundary to `layer4_paired_compressors`, with
  119 terminal dispatches, 61/61 no-copy mappings, and six pinned checksums.

Validation:

- Fixture verifier: valid six-tensor, 1,392,640-byte differential fixture.
- Rust unit suite: 100 passed.
- Focused optimized M1 Ultra run: all 1,024 emissions and four recurrent-state
  tensors C0 exact, 740.517 ms wall / 656.454 ms GPU.
- Full target-Mac gate: 100 Rust tests and 61 Python tests passed, every native
  control remained exact, and the new terminal command reported 750.548 ms
  wall / 655.250 ms GPU with 61/61 no-copy mappings.

Next:

- Complete layer-4 dense mixed attention and FFN from the retained state.

### 2026-08-22 — Complete layer 3 flows into exact layer-4 Q/KV state

Objective:

- Continue the exact native 2K chain from layer 3's retained final HC through
  layer 4's complete pre-compressor Q/KV boundary.

Changes:

- Captured layer 4's HC attention ingress, learned norm, Q-Lora, fused Q/KV
  norm, Q-B, Q/KV RoPE, and FP8 KV-finalization tensors in four fresh DwarfStar
  processes. The separately captured 256 MiB Qraw tensors and every other full
  tensor were byte-identical between repeats.
- Added `prefill-layer4-qkv-2048-v1`, retaining ten exact final tiles while
  pinning all complete 2K identities by SHA-256.
- Added nine no-copy layer-4 model spans and ten Metal dispatches to the
  terminal command. The runtime retains the full layer-4 Q, KV, input HC, and
  attention split buffers for its compressor/attention continuation.
- Extended the ABI, C0 checks, stable JSON report, documentation, and fixture
  shape coverage through the `layer4_KVcur` boundary.

Validation:

- Focused optimized M1 Ultra run: all ten boundaries C0 exact, 89 dispatches,
  53/53 no-copy model mappings, 1169.303 ms wall / 647.777 ms GPU.
- Complete target-Mac gate: 99 Rust tests and 61 Python tests passed, every
  differential fixture verified, all retained Metal controls stayed exact,
  and the new boundary repeated C0 exact at 947.998 ms wall / 651.830 ms GPU.
- These timings include correctness setup/readback and are not throughput
  claims.

Next:

- Drive layer 4's paired ratio-4 attention/indexer compressors from the
  retained normalized activations and validate all 512 prompt emissions.

### 2026-08-22 — Native 2K prefill is complete through layer 3

Objective:

- Continue the exact full-2K layer-3 attention state through its complete FFN
  and additive four-stream HC post-state without a host activation handoff.

Changes:

- Captured the ten layer-3 FFN oracle boundaries twice in fresh DwarfStar
  processes. Every full tensor was byte-identical; the compact checked fixture
  retains all final 32-row boundaries and pins every full-2K SHA-256 identity.
- Added a batch Metal top-k router that reproduces DwarfStar's sequential
  strict-`>` top-6 selection over `sqrt(softplus(logits)) + bias`. Expert
  weights are gathered from the unbiased probabilities before normalization
  and the 1.5 scale.
- Extended the terminal command through FFN HC ingress, router projection,
  routed IQ2_XXS gate/up plus Q2_K down, the Q8_0 shared expert, and the final
  additive HC update. Large routed/shared scratch allocations are reused only
  after layer 2 completes.
- Added twelve layer-3 FFN model spans, seven ABI outputs, exact final-tile C0
  checks, a full-final-HC checksum guard, stable JSON metadata, and a complete
  layer-3 prefill claim.

Validation:

- Two fresh oracle captures agreed bit-for-bit for all 2,048 rows, including
  all biased top-k selections and the complete 128 MiB final HC tensor.
- Focused optimized M1 Ultra run: C0 exact, 79 dispatches, 44/44 no-copy model
  mappings, 675.511 ms wall / 592.734 ms GPU.
- Complete target-Mac gate: 98 Rust tests and 61 Python tests passed, every
  differential fixture verified, all retained Metal controls remained exact,
  and the new boundary repeated C0 exact at 759.645 ms wall / 671.882 ms GPU.
- These timings include correctness-oriented execution/readback and are not a
  throughput claim.

Next:

- Continue the native 2K prefill chain through layer 4, then implement sparse
  indexed attention beyond the fixed 512-row dense ratio-4 prompt boundary.

### 2026-08-22 — Layer 3 owns dense mixed attention and additive HC post

Objective:

- Continue the exact native layer-3 batch from its ratio-128 compressor through
  dense mixed attention, inverse RoPE, both output projections, and additive HC
  post-state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical layer-3 `attn_out` and
  `hc_attn_post` captures. Their full SHA-256 identities are
  `5bfadcbc1d2ee7b42753b506420045409ba277ae2ffc7dbd0e87187e51b74e13`
  and `47c26665144097e0912284961d95f9b3ae72c8ce40c271e028dfdf71f4ee453b`.
- Independent repeated `kqv_out`, `kqv_back`, and `attn_low` captures pin row-0
  diagnostics for the attention and output-projection stages.
- `prefill-layer3-attention-2048-v1` retains the complete 32 MiB attention
  output, the final 32-row HC tile, and the three row-0 diagnostic tensors. The
  full 128 MiB HC identity is pinned without retaining a duplicate large blob.

Implementation:

- Added three mmap-backed model views for layer-3 sinks and Q8 attention output
  weights, plus nine ordered Metal dispatches for raw/compressed F16 staging,
  mask block scan, FlashAttention, inverse RoPE, grouped low projection, dense
  output projection, and four-stream HC expansion.
- The 2,064 logical keys are physically padded to 2,112 rows with 48 fully
  masked entries. This satisfies the non-vector FlashAttention specialization's
  64-row block contract; the prior 2,560-key layer-2 path was already aligned
  and therefore did not expose the requirement.
- Extended the C ABI, Rust report/JSON contract, fixture importer, persistent
  layer-3 HC ownership, and diagnostic comparisons. The terminal checkpoint now
  uses 58 dispatches and 32/32 no-copy model mappings.

Validation:

- The optimized real-model probe reproduced all 8,388,608 layer-3 attention
  values and the final 524,288-value HC tile bit-for-bit while preserving every
  prior boundary. It reported 583.424 ms wall / 501.227 ms GPU with correctness
  readback in scope; this is not a throughput measurement.
- The complete target-Mac gate passed 97 Rust tests and 61 Python tests, every
  fixture verifier and retained Metal control, and a second exact layer-3
  attention run at 569.588 ms wall / 490.821 ms GPU. The 2K sequential control
  remained C0 exact at 21.958 tokens/s over 93270.010 ms.

Decision and next:

- Accept `layer3_attention_hc_post` as the next exact native-batch checkpoint.
  Continue through the layer-3 biased top-k FFN and additive FFN HC post-state.

### 2026-08-22 — Layer 3 owns all 16 ratio-128 prompt emissions

Objective:

- Continue the exact native layer-3 batch from retained full-2K Q/KV state
  through the complete ratio-128 attention compressor and persistent state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical `KVcompress`,
  `attn_state_kv`, and `attn_state_score` captures at layer 3.
- The 16x512 compressed output has SHA-256
  `9e6d904dc6df0601d0b3c32f9baf58230c893346d050b9330d5c38cc89143481`.
  The two 128x512 states have SHA-256 identities
  `8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90`
  and `6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778`.
- `prefill-layer3-compressor-2048-v1` retains all three complete tensors; no
  output is truncated to a final tile.

Implementation:

- Added two full-batch F16 compressor projections, exact score/APE addition,
  DwarfStar-order scalar 128-row softmax pooling, weighted RMSNorm, compressed
  YaRN RoPE, and E4M3FN finalization to the existing terminal command.
- Added four strict no-copy model views and retained the 16 compressed rows and
  both recurrent-state buffers in the persistent Metal context.
- Extended the C ABI, Rust report/JSON contract, fixture importer, and
  Rust/Python fixture tests. The checkpoint now uses 49 dispatches and 29/29
  no-copy model mappings.

Validation:

- The optimized real-model probe matched all three new full tensors bit-for-bit
  while preserving every prior layer-2 and layer-3 boundary. It reported
  494.638 ms wall / 396.686 ms GPU with correctness readback in scope; this is
  not a throughput measurement.
- The complete target-Mac gate passed 96 Rust tests and 61 Python tests, every
  fixture verifier and retained Metal control, and a second exact compressor
  run at 483.530 ms wall / 399.101 ms GPU. The 2K sequential control remained
  C0 exact at 21.939 tokens/s over 93347.958 ms.

Decision and next:

- Accept `layer3_ratio128_compressor` as the next exact native-batch checkpoint.
  Continue through layer-3 dense mixed attention and its additive HC post-state
  before implementing the biased top-k FFN.

### 2026-08-22 — Layer 3 owns exact full-2K Q/KV state

Objective:

- Continue the exact native layer-3 batch from Q-Lora through Q/KV setup,
  compressed RoPE, FP8 finalization, and persistent raw-KV ownership.

Oracle evidence:

- Two fresh primary DwarfStar processes produced byte-identical full-2K
  `q_lora_norm`, `KVraw`, `KVnorm`, `Qcur`, `KVrope`, and `KVcur` captures.
  Two additional fresh processes produced identical 256 MiB `Qraw` captures.
- The repeated `q_lora` capture retained SHA-256
  `bafe82a1535457caec52278bdb3c95e317aacb128c94f704fc523ca9581d6265`,
  exactly matching the prior checkpoint.
- `prefill-layer3-kv-state-2048-v1` retains exact final tiles for all seven new
  boundaries and pins their full capture SHA-256 identities.

Implementation:

- Added six dispatches to the existing terminal command: Q8 KV projection,
  fused Q/KV learned norm, Q8 Q-B, Q head norm/RoPE, KV RoPE, and FP8 KV
  finalization.
- Added four no-copy model views and retained the complete layer-3 finalized
  KV, Q, input HC, and attention split buffers in the persistent Metal context.
- Extended the ABI, strict tensor checks, stable JSON, fixture importer, and
  Rust/Python fixture tests. Large Qraw/Qcur identities use exact final tiles;
  the manifest pins their complete 2K SHA-256 captures.

Validation:

- The optimized focused probe matched all seven new final-tile boundaries
  bit-for-bit and matched the full-2K checksums for Q-Lora norm, KVraw, KVnorm,
  KVrope, and KVcur.
- The terminal schedule used 42 dispatches and 25/25 no-copy mappings. It
  reported 482.230 ms wall / 402.931 ms GPU with correctness readback in scope;
  this is not a throughput measurement.
- The complete target-Mac gate passed 95 Rust tests and 60 Python tests, every
  fixture verifier and retained Metal control, and a second exact layer-3 run
  at 447.543 ms wall / 366.771 ms GPU. The 2K sequential control also remained
  C0 exact at 20.168 tokens/s over 101546.809 ms.

Decision and next:

- Accept `layer3_kv_cur` as the next exact native-batch checkpoint. Build the
  layer-3 ratio-128 compressor state and its 16 prompt emissions before dense
  mixed attention and the distinct biased top-k FFN.

### 2026-08-22 — Full-2K layer-2 HC flows directly into layer-3 Q-Lora

Objective:

- Cross the first native layer-3 batched-prefill boundary without uploading or
  reconstructing layer 2's completed 16,384-wide HC state on the host.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical full-2K captures for
  layer 3 `hc_attn_pre`, `attn_norm`, and `q_lora`.
- Their full capture SHA-256 values are respectively
  `615f67cb9738b583263e1a7abd3970ad4df818f0bc9220053be4cbc6ba7d6cab`,
  `1348e3368a4c6b7185e730f11545c74a7fe8fa1c9877fcc9200a5de405f2b818`,
  and `bafe82a1535457caec52278bdb3c95e317aacb128c94f704fc523ca9581d6265`.
- `prefill-layer3-ingress-2048-v1` retains the exact final 32 rows for all
  three boundaries and pins the full captures in its manifest.

Implementation:

- Extended the existing terminal layer-2 command with four dispatches: HC RMS
  norm, the F16 HC mixer, fused HC collapse plus learned attention norm, and
  the Q8 Q-A projection.
- Added five mmap-backed layer-3 model views and kept the complete layer-2 HC
  buffer GPU-resident across the boundary.
- Extended the Rust ABI, exact tensor-shape validation, JSON evidence, fixture
  importer, fixture verifier coverage, and focused unit tests.

Validation:

- The optimized real-model probe passed on the M1 Ultra with all three new
  final-tile tensors bitwise equal to DwarfStar and their full-2K FNV checksums
  equal to `0xcb93e0a251fd7280`, `0xe64fba2f04bfcb54`, and
  `0xd01875758ab4b722`.
- The terminal schedule used 36 dispatches and retained 21/21 no-copy model
  mappings. It reported 441.440 ms wall / 371.934 ms GPU with correctness
  readback in scope; this is not a throughput measurement.
- The complete target-Mac gate passed: 94 Rust tests, the optimized build,
  Metal ownership/dispatch validation, all differential
  fixtures, the Rust/Python artifact contract, every real-model Metal control,
  and the 2K sequential diagnostic. The gate repeated the new boundary at
  456.961 ms wall / 376.116 ms GPU.
- After adding the fixture-specific Python catalog assertion, the final
  platform-independent regression pass contained 59 Python tests; all passed.

Decision and next:

- Accept `layer3_q_lora` as the next exact native-batch checkpoint. Continue
  through layer 3's Q/KV setup and state ownership before implementing its
  distinct biased top-k FFN.

### 2026-08-16 — Complete layer 2 reaches the exact full-2K HC frontier

Objective:

- Continue the exact layer-2 mixed-attention boundary through both HC updates,
  token-hash routing, routed/shared experts, and the final 16,384-wide HC state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical full-2K captures for
  `hc_attn_post`, FFN ingress/norm/router/expert boundaries, and `hc_ffn_post`.
- The full attention-HC state has SHA-256
  `4f7c61ad617347f186cb959457d5c6f1c95692451bf186c751f863a6417baad2`;
  the full final-HC state has SHA-256
  `67dbac97346ee6bea6bb967eafa6c7841fb3477798f9cc154c9dfed24ff5564b`.
- `prefill-layer2-complete-2048-v1` retains exact final-tile payloads and pins
  every full capture identity without adding a single 128 MiB fixture blob.

Implementation:

- The 64-tile context now retains every layer-2 input HC row, attention split,
  and token ID alongside Q/KV and compressor state.
- The terminal Metal command adds the attention HC expand and a full 2,048-row
  FFN batch: HC ingress/learned norm, token-hash router, fused IQ2_XXS
  pair-SwiGLU, Q2_K routed down/sum, Q8_0 shared expert, and additive HC expand.
- The decomposed router kernels' explicit maximum grew from 32 to 2,048 rows;
  the existing 32-row paths preserve identical dispatch sizes and arithmetic.

Validation:

- The full attention output, attention-HC final tile, FFN ingress, router
  selection/weights, routed/shared outputs, and final-HC final tile all match
  DwarfStar by FP32 bit pattern.
- Full-state checksums match both 2K HC capture identities. The terminal command
  retains 16/16 no-copy model mappings and uses 32 dispatches.
- The focused M1 Ultra run reported 408.397 ms wall / 338.484 ms GPU, including
  synchronization and correctness readback; it is not a throughput claim.
- The complete target-model runtime gate, 58 Python tests, all Rust tests, every
  pinned fixture verifier, and all prior Metal decode/prefill controls pass.

Next:

- Carry the final layer-2 HC state into the first exact layer-3
  batched-prefill boundary while retaining this complete layer-2 control.

### 2026-08-16 — Exact layer-2 dense mixed attention reaches the full 2K frontier

Objective:

- Continue the exact empty-seed layers-0/1/2 prefill path through layer 2's
  first real attention consumption boundary.

Oracle evidence:

- Tracing the pinned DwarfStar path corrected an important boundary
  assumption: exactly 512 ratio-4 compressed rows still use dense mixed
  attention. Sparse indexer top-k begins only when the compressed cache grows
  beyond 512 rows, after the 2K prompt.
- Two fresh-process full-2K attention-output captures were byte-identical.
- The retained 2048 x 4096 FP32 payload is 33,554,432 bytes with SHA-256
  68c2110283b472105f00e192817dbb682ebe0815f2ba4a76cd73ed20f3d97508.
- prefill-layer2-attention-2048-v1 records the exact 2,048 raw plus 512
  compressed KV shape and the absence of sparse selection at this boundary.

Implementation:

- The persistent tile context now retains all 2,048 layer-2 normalized query
  rows alongside the already exact raw KV and paired-compressor state.
- rust_star_metal_run_prefill_layer2_attention maps Q-B, attention sinks, and
  both output tensors once without copying, then executes a ten-dispatch
  command: full Q8 Q-B, compressed YaRN head norm/RoPE, raw and compressed F16
  staging, dense mixed FlashAttention, inverse compressed RoPE, and
  grouped/dense Q8 output projection.
- The mixed mask preserves the 128-row raw causal window and exposes
  (query + 1) / 4 compressed rows.
- prefill-layers012-attention-loop-probe reuses the exact 64-tile compressor
  loop in the same Metal context and compares all 8,388,608 output values by
  FP32 bit pattern.

Validation:

- The first optimized M1 Ultra run matched the complete DwarfStar attention
  output bit-for-bit.
- The isolated attention command retained 4/4 no-copy model mappings, used ten
  dispatches, and reported 196.069 ms wall / 169.211 ms GPU. This interval
  includes correctness-oriented setup, synchronization, and full readback; it
  is not a prefill-throughput claim.
- The fixture verifier, optimized build, all 92 Rust tests, all 58 Python
  artifact tests, and the complete target-model Metal runtime gate pass.

Scope:

- Layer 2 now has exact full-2K raw KV, both ratio-4 compressors, dense mixed
  attention, inverse RoPE, and attention output projection.
- Layer-2 attention HC post-processing, FFN, complete layer-2 prefill, later
  layers, output logits, and sparse ratio-4 attention after the prompt remain
  pending.

Next:

- Continue the live full-2K boundary through layer 2's attention HC update and
  FFN, then decide whether to advance batched prefill layer-by-layer or isolate
  the first 513-to-512 sparse selection in post-prompt decode.

### 2026-08-16 — Layer-2 paired compressors reach the full 2K frontier

Objective:

- Extend the empty-seed native layers-0/1 and layer-2 raw-KV loop through both
  ratio-4 compressors without introducing a host activation handoff.

Oracle and implementation:

- Captured the complete layer-2 attention/indexer compressed caches and final
  recurrent KV/score states twice in fresh DwarfStar processes. All six payload
  pairs were byte-identical. Added the strict
  `prefill-layer2-compressors-2048-v1` fixture, its provenance-preserving
  importer, and fixture-shape tests.
- Expanded the persistent Metal context from 57 to 65 mmap-backed model ranges
  and added owned attention/indexer compressed caches plus recurrent states.
  Four aligned F16 projections feed DwarfStar-equivalent fused eight-row
  pooling, learned norm, compressed RoPE, and E4M3FN/indexer QAT.
- Preserving DwarfStar arithmetic required two distinct details. The one-shot
  2K capture pools the aligned compressor rows as one fused batch rather than
  repeating the legacy one-row reduction. Regular tiles keep batch-projected
  tail state so later rows remain identical to that one-shot capture; only the
  final tile reruns its four-row tail with the small-batch projection used by
  DwarfStar's published final recurrent state.
- Added `prefill-layers012-compressor-loop-probe`, a stable JSON schema, atomic
  CLI output, exact prefix/state validation, and the command to the target-Mac
  validation script. Regular tiles use 118 dispatches; the final tail refresh
  uses 122.

Target-Mac evidence:

- The focused full-2K command passed all 512 attention compressed rows, all 512
  indexer compressed rows, and the four final recurrent state tensors by FP32
  bit pattern. Every tile preserved 65/65 no-copy mappings. Summed diagnostic
  intervals were 3624.704 ms wall and 3514.757 ms GPU; setup, synchronization,
  and exhaustive correctness checks make these non-throughput measurements.
- The pinned fixture verifier accepted 8 operations, 6 tensors, and 1,392,640
  payload bytes. Formatting and 90 Rust tests passed after adding the command
  and report contracts.
- The complete target-Mac gate repeated the new command at 3548.253 ms summed
  wall and 3444.137 ms summed GPU, then passed all retained runtime controls.
  The gate included 90 Rust tests, 57 Python tests, all 242 pinned differential
  fixtures, an optimized macOS build, strict validation of all 1,288 required
  model tensors, and both steady-state benchmark controls.

Decision and next:

- Accept exact persistent ownership of both layer-2 ratio-4 compressors across
  native 2K prefill. Next consume the resulting 512-row indexer cache for exact
  selection and mixed raw/compressed attention, then cross the layer-2 FFN.
- This checkpoint makes no claim about layer-2 mixed attention, layer-2 FFN,
  complete-model prefill, or inference throughput.

### 2026-08-16 — Layer-2 raw KV ownership reaches the full 2K frontier

Objective:

- Continue the exact layer-2 normalized-KV seam through its compressed-attention
  RoPE, E4M3FN finalization, and retained raw-KV state.

Oracle and implementation:

- Captured full-2K layer-2 `KVrope` and `KVcur` twice in fresh DwarfStar
  processes. Each corresponding 4 MiB pair was byte-identical. Their SHA-256
  values are `d46da14951b304fb4a19be43b82d350b273337de042da2308530438e431e117d`
  and `07f19c5197442f3c85350b32d0661e81b3f105a0e8640d3b3bced6c333267135`.
- Added the strict `prefill-layer2-kv-state-2048-v1` fixture/importer and the
  `prefill-layers012-kv-state-loop-probe` command. Layer 2 uses the production
  YaRN constants and E4M3FN simulation, then appends each finalized tile into
  one persistent GPU allocation.
- Every continuation compares the retained layer-2 prefix against the oracle.
  The extended tile uses 92 dispatches and 57 mmap-backed model views.

Target-Mac evidence:

- The focused full-2K loop passed all 64 `KVnorm`, `KVrope`, and `KVcur` tile
  comparisons plus all accumulated layer-0/layer-1/layer-2 KV-prefix gates.
  Summed correctness-oriented intervals were 3605.869 ms wall and 3497.958 ms
  GPU.
- Formatting, compilation, 88 Rust tests, and 32 focused Python fixture tests
  passed before the full repository gate.
- The complete target-Mac gate then passed 88 Rust tests, all 57 Python tests,
  all 241 pinned differential fixtures, and strict validation of all 1,288
  required model tensors. Its independent layer-2 KV-state replay reported
  3600.907 ms wall and 3497.247 ms GPU, preserved 57/57 no-copy model mappings
  for every tile, and was followed by every retained decoder control and
  steady-state benchmark.

Decision and next:

- Accept the complete native layer-2 raw-KV state as C0 exact ownership
  evidence, not throughput. Next implement the paired ratio-4 attention and
  indexer compressors, then cross mixed raw/compressed attention and the FFN.

### 2026-08-16 — Every live layer-1 output reaches exact layer-2 KVnorm

Objective:

- Use a compact downstream layer-2 oracle to validate every post-FFN layer-1
  output produced by the full native 2K loop.

Oracle and implementation:

- Captured layer 2's full `KVnorm` tensor twice from fresh DwarfStar processes.
  Both 4 MiB payloads were byte-identical with SHA-256
  `089138d8fc82c1eb55754451707f59475f2afb2a356dcc505314ddf29814e7b6`.
- Added the strict `prefill-layer2-kvnorm-2048-v1` fixture/importer and the
  `prefill-layers012-kvnorm-loop-probe` command. Each of the 64 tiles now adds
  layer-2 HC RMSNorm, F16 HC projection, fused HC collapse/learned norm, Q8_0
  Q-A and KV projection, and fused Q/KV learned RMSNorm.
- The new schedule uses 90 dispatches and 57 mmap-backed no-copy model views per
  tile. Each `32x512` output slice is compared bit-for-bit before continuation.

Target-Mac evidence:

- The focused full-2K loop passed all 64 layer-2 slices and every retained
  layer-0/layer-1 KV-prefix check. Summed correctness-oriented intervals were
  3688.567 ms wall and 3593.970 ms GPU.
- The complete gate repeated the new loop at 3543.522/3450.060 ms wall/GPU,
  then passed formatting, all 86 Rust tests, 56 Python tests, all 240 pinned
  differential fixtures, optimized Objective-C/Metal compilation, all 1,288
  required model tensors, every established Metal/decoder control, and both
  steady-state benchmarks.

Decision and next:

- Accept every live layer-1 output row as downstream C0 validated through the
  native layer-2 normalized-KV seam. Do not treat these summed intervals as
  throughput. Next cross layer 2's compressed RoPE/cache/attention boundary and
  finish its FFN before generalizing the full native prefill layer loop.

### 2026-08-16 — Empty-seed layers-0/1 KV ownership reaches all 2K prompt rows

Objective:

- Turn the two-tile persistent checkpoint into a reusable loop from position 0
  through 2047 without introducing another captured execution prefix.

Oracle reuse and implementation:

- Reused the existing 2,048 canonical token IDs plus the captured layer-0 and
  layer-1 KV prefixes through row 2015. No new large fixture was required.
- Generalized native reset/retain/consume state from the tail pair to every
  aligned 32-row tile. Tile 0 starts with zero captured KV rows; each of the 63
  continuations requires contiguous retained state and compares both complete
  accumulated prefixes against the oracle before appending.
- Added `prefill-layers01-live-kv-loop-probe` with schema
  `rust-star-prefill-layers01-live-kv-loop-probe-v1`. Its stable artifact
  records all 64 tile schedules, retained row counts, mapping identity, timing,
  and the exact claim boundary.
- The final tile still performs exhaustive layer-0/layer-1 output comparison,
  and the independently retained positions 1984--2015 tile remains fully C0
  checked. Other non-final layer-1 post-FFN outputs execute but are not all
  retained, so the artifact explicitly denies a complete layers-0/1 or model
  prefill claim.

Target-Mac evidence:

- The focused full loop advanced empty KV buffers to 2,048 exact rows for both
  layers. All 64 tiles preserved 49/49 mmap-backed model views; summed
  correctness-oriented intervals were 3492.491 ms wall and 3393.021 ms GPU.
- The complete gate repeated the 64-tile result at 3489.173/3383.418 ms
  wall/GPU, then passed formatting, all 84 Rust tests, 55 Python tests, all 239
  differential fixtures, optimized Objective-C/Metal compilation, all 1,288
  required model tensors, every established Metal/decoder control, and both
  steady-state benchmarks.

Decision and next:

- Accept the complete 2K layer-0/layer-1 KV chain as exact state-ownership
  evidence, not as a throughput or complete-prefill result. Next extend native
  batched execution through layer 2 and use the layer-2 KV oracle as the
  downstream validation boundary for live layer-1 outputs.

### 2026-08-16 — Live layer-0/layer-1 KV state crosses the tile boundary

Objective:

- Replace the captured KV boundary between the two exact complete layers-0/1
  tiles with Rust-owned, device-resident state while preserving the independent
  captured-prefix replay as a regression control.

Implementation:

- Added explicit reset/retain/consume modes to the native prefill boundary and
  persistent layer-0/layer-1 full-KV buffers to one Metal context.
- The positions 1984--2015 tile initializes from its 1,984-row oracle prefix,
  appends both live 32-row KV tiles, and retains 2,016 rows per layer. The
  positions 2016--2047 tile requires that contiguous state, validates both
  retained prefixes against the C0 oracle, then appends and attends from the
  retained buffers rather than copying its captured prefix into execution
  storage.
- Added `prefill-layers01-live-kv-chain-probe` and stable schema
  `rust-star-prefill-layers01-live-kv-chain-probe-v1`. Its artifact explicitly
  records one persistent context, two command buffers, one inter-tile host
  wait, a captured first prefix, no captured final execution prefix, and no
  full-prefill or single-command-buffer claim.
- Kept `prefill-layers01-row-coverage-probe` unchanged as the independent
  captured-prefix-per-tile control and added both commands to the Mac gate.

Target-Mac evidence:

- The focused live chain passed both tiles bit-for-bit with 49/49 no-copy model
  views per tile. It retained 2,016 rows after the first tile and 2,048 after
  the second; the focused intervals were 129.589/90.751 and 52.422/51.523 ms
  wall/GPU.
- The independent captured-prefix control also remained exact at
  129.548/90.039 and 127.754/87.312 ms wall/GPU.
- The complete gate passed formatting, all 83 Rust tests, 55 Python tests, all
  239 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The live first/final
  tiles repeated exact at 129.000/90.011 and 52.492/51.516 ms wall/GPU.

Decision and next:

- Accept persistent two-tile layer-0/layer-1 KV ownership as the next native
  prefill checkpoint. Next turn the pair into a reusable backward-extending
  tile loop, preserving per-tile C0 evidence while avoiding a premature full
  2K or throughput claim.

### 2026-08-16 — Exact layers-0/1 coverage expands to two prompt tiles

Objective:

- Remove the final-tile-only position and cache-row assumptions, then prove the
  complete layers-0/1 schedule at the immediately preceding M1 tile while
  preserving every shorter boundary as an independent control.

Oracle evidence and fixture:

- Reused full layer-0 and layer-1 DwarfStar captures from two fresh 2K
  processes. The layer-0/layer-1 `KVcur`, `hc_ffn_post`, and
  `ffn_moe_topk` full-tensor pairs were byte-identical and retained their
  previously pinned SHA-256 identities.
- Added `prefill-layers01-previous-tile-2048-v1`, a compact decisive fixture
  for positions 1984--2015 containing six tensors and 4,326,912 verified bytes.
  It retains both layers' KV tile, selected expert IDs, and final four-stream
  HC state without duplicating every large intermediate.

Implementation:

- Generalized the 84-dispatch complete layers-0/1 executor from fixed position
  2016 to any aligned 32-row tile within the 2K frontier. KV-prefix length,
  RoPE positions, attention visibility, raw-ring target row, and guard length
  now derive from `position_start`.
- Added `prefill-layers01-row-coverage-probe` with schema
  `rust-star-prefill-layers01-row-coverage-probe-v1`. It executes positions
  1984--2015 and the preserved 2016--2047 control as two complete native tiles,
  each with 49 direct mmap-backed model views.
- Stable JSON explicitly records that the two tiles use captured per-tile KV
  prefixes and therefore does not claim a live inter-tile KV chain or complete
  native prefill.

Target-Mac evidence:

- The warm focused probe passed both tiles C0 exact at 127.196/87.020 ms and
  124.717/87.421 ms wall/GPU, with 49/49 no-copy views for each tile.
- The complete gate passed formatting, all 82 Rust tests, 55 Python tests, all
  239 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The earlier/final tiles
  remained C0 exact at 128.771/91.582 and 124.334/87.370 ms wall/GPU.

Decision and next:

- Accept arbitrary-position exact 32-row replay and 64 covered prompt rows as
  the next row-coverage checkpoint. Next replace the captured boundary between
  these two tiles with live per-layer KV ownership before scaling the tile loop.

### 2026-08-15 — Native M1 final tile completes layer 1

Objective:

- Extend the direct live layer-0-to-layer-1 HC handoff through the remainder of
  uncompressed layer 1 while preserving the 43-dispatch layer-0 and
  47-dispatch layer-1-Q-A controls as independent commands.

Oracle evidence and fixture:

- Captured twenty full 2K layer-1 tensors twice from fresh pinned DwarfStar
  processes. Every pair was byte-identical. The strict importer pins every
  full-capture SHA-256 before retaining the final 32 rows plus the 2,016-row KV
  prefix needed for rectangular attention.
- Added `prefill-layer1-complete-2048-v1`: nine operations, twenty-one tensors,
  and 26,543,616 verified bytes. The fixture covers Q/KV setup and RoPE, KV
  finalization, FlashAttention, grouped attention output, both HC updates, the
  decomposed token-hash router, routed experts, and the shared expert.

Implementation:

- Added `prefill-layers01-complete-boundary-probe` with schema
  `rust-star-prefill-layers01-complete-boundary-probe-v1`. The optional Metal
  continuation adds the remaining thirty-seven layer-1 dispatches after Q-A,
  for 84 total: 43 in layer 0 and 41 in layer 1.
- The same command buffer consumes the live layer-0 four-stream FFN HC state,
  reconstructs layer 1's 2,048-row contiguous KV input from the captured prefix
  and live final tile, and finishes at `layer1_hc_ffn_post` without a host
  activation seam. All 49 model ranges are direct mmap-backed views.
- The complete command retains and checks 12,878,208 produced FP32 values plus
  384 selected expert IDs across both layers. Stable JSON asserts the complete
  layer-1 final-tile claim while continuing to reject a full-prefill claim.

Target-Mac evidence:

- A focused optimized run preserved the 43-dispatch 25/25-view and 47-dispatch
  30/30-view controls C0 exact, then passed the new 84-dispatch boundary C0
  exact with 49/49 no-copy views at 243.211 ms wall / 192.411 ms GPU.
- The complete gate passed formatting, all 80 Rust tests, 54 Python tests, all
  238 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The new boundary remained
  exact at 277.821 ms wall / 226.582 ms GPU; the preserved 43- and 47-dispatch
  controls remained exact at 131.947/76.576 and 129.940/78.788 ms wall/GPU.

Decision and next:

- Accept complete native layers 0 and 1 for the isolated final M1 tile. Next
  broaden native retained-row coverage toward the complete 2K prefill rather
  than adding another layer to the isolated tile.

### 2026-08-15 — Native final tile hands layer-0 HC directly into layer-1 Q-A

Objective:

- Preserve the complete exact layer-0 final-tile command as an independent
  control while proving its live four-stream FFN HC output can feed native
  layer-1 batch arithmetic without synchronization or a host activation seam.

Oracle evidence and schedule decision:

- Selected the smallest decisive layer-1 boundary: plain four-stream RMSNorm,
  legacy F16 HC mixer, fused HC split/collapse/learned attention norm, and the
  legacy Q8_0 Q-A batch projection.
- Captured full layer-1 `hc_attn_pre`, `attn_norm`, and `q_lora` tensors from
  two fresh pinned DwarfStar 2K processes. All three pairs were byte-identical.
  Their full SHA-256 values are
  `a5ad89aaa9a5c3537c22a26730b918bdde4e07177e72c643059306bbb28439bc`,
  `37b0f1a783c0968445dd77214b2f62b62a0f5dcac01778ca5e0627948c392571`,
  and `ff0a9a83d0f3077c83ad1ba223310a496d3d8b51709da986d97887f4b0836901`.

Implementation:

- Added `prefill-layer1-ingress-2048-v1` and a reproducible strict importer
  that binds both full capture sets before retaining positions 2016--2047.
  The fixture contains three tensors and 1,179,648 verified bytes.
- Added `prefill-layers01-boundary-probe` with schema
  `rust-star-prefill-layers01-boundary-probe-v1`. The narrow Metal ABI accepts
  an optional layer-1 continuation, so the existing v5 layer-0 command still
  stops at exactly 43 dispatches and 25 model views.
- When requested, the same encoder feeds `after_ffn_hc_buffer` directly into
  four layer-1 dispatches and five additional no-copy model views. The new
  boundary retains 7,274,688 produced FP32 values plus the layer-0 router's 192
  selected expert IDs, reports the direct handoff in stable JSON, and keeps
  complete-layer-1 and full-prefill claims false.

Target-Mac evidence:

- The focused optimized run first repeated the preserved v5 control C0 exact
  at 124.696 ms wall / 65.868 ms GPU, then passed the 47-dispatch layers-0/1
  boundary C0 exact with 30/30 no-copy views at 125.125 ms wall / 69.515 ms
  GPU. Both include setup, synchronization, exhaustive readback, and comparison.
- The complete gate passed formatting, all 78 Rust tests, 53 Python tests, all
  237 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. Its layers-0/1 boundary
  remained exact at 129.903 ms wall / 78.876 ms GPU. The 2K sequential
  diagnostic remained exact against its replay oracle and preserved the
  required native-batch mismatch and paired-run ineligibility.

Decision and next:

- Accept the direct live HC handoff and layer-1 Q-A output as the next exact
  native boundary. Extend it through the remaining uncompressed layer-1
  attention and FFN schedule before broadening retained row coverage.

### 2026-08-15 — Native M1 final tile completes layer 0 through the FFN HC tail

Objective:

- Extend the continuous final 32-row token-to-attention-HC boundary through
  layer 0's active M1 FFN router, routed/shared experts, and additive HC tail.

Oracle evidence and schedule decision:

- Traced the pinned M1 production path through legacy F16 FFN ingress/router,
  decomposed softplus/sqrt/token-hash routing, expert-major mapping, fused
  IQ2_XXS paired gate/up weighted SwiGLU with an F16 intermediate, Q2_K routed
  down projection, six-expert sum, three Q8_0 shared projections with flat
  SwiGLU, and additive `kernel_dsv4_hc_expand4`.
- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/weights,
  routed weighted-SwiGLU/output, shared output, and `hc_ffn_post` from two fresh
  pinned 2K DwarfStar processes. All ten full payload pairs were byte-identical.
  The strict importer binds their exact sizes and full SHA-256 identities
  before retaining positions 2016--2047.
- DwarfStar graph capture disables the fused routed pair for observability, but
  its production source contract requires the retained F16 intermediate and
  routed output to be bit-identical. Rust Star executes the production fused
  kernel and compares those outputs to the retained diagnostic boundary.

Implementation:

- Added `prefill-ffn-output-2048-v1`, its reproducible strict importer, Rust
  shape checks, and Python manifest/payload validation.
- Evolved `prefill-layer0-boundary-probe` to schema v5. One Rust-owned command
  buffer now runs 43 compute dispatches continuously from the final 32 token
  IDs through `hc_ffn_post`. The boundary wraps all 25 GGUF weight ranges as
  read-only no-copy Metal views and compares 6,979,776 produced FP32 values by
  bit pattern plus 192 selected expert IDs exactly.
- Added the seven decomposed M1 batch-router kernels and wired the native
  expert map, fused IQ2_XXS pair, Q2_K F16 down, sum-six, shared SwiGLU, and HC
  tail kernels behind the narrow Rust/C ABI. Stable JSON now reports both the
  attention and FFN oracle fixtures, the full schedule, retained checksums, and
  an explicit complete-layer-0-final-tile claim while preserving
  `full_prefill_claim: false`.

Target-Mac evidence:

- The focused optimized v5 run was C0 exact with 43 dispatches and 25/25
  no-copy views, reporting 135.893 ms wall and 73.494 ms GPU. Setup,
  synchronization, exhaustive readback, and comparison remain in scope; this
  is correctness evidence, not throughput.
- The complete gate passed formatting, all 76 Rust tests, 52 Python tests, all
  236 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. Its v5 run remained C0
  exact at 138.441 ms wall / 81.329 ms GPU with 25/25 no-copy views. The 2K
  sequential diagnostic remained exact against its replay oracle while
  preserving the required 129,280-logit native-batch mismatch and paired-run
  ineligibility.

Decision and next:

- Accept the FFN/MoE/HC suffix as the exact completion of layer 0 for the
  isolated final native tile. Next hand its live HC state into layer 1 and
  broaden row coverage; do not relabel the final-tile gate as full 2K prefill.

### 2026-08-15 — Continuous native M1 final tile crosses attention output and HC post

Objective:

- Extend the exact final 32-row token-to-`kqv_back` boundary through layer 0's
  active grouped Q8 attention output and four-stream HC post-update.

Oracle evidence and schedule decision:

- Traced the M1 path rather than assuming the Metal 4 schedule. On this machine
  TensorOps is unavailable, `ds4_gpu_attention_output_q8_batch_f16_tensor`
  returns unsupported, and DwarfStar therefore uses
  `kernel_mul_mm_id_map0_ne20_8`, `kernel_mul_mm_id_q8_0_f32`,
  `kernel_mul_mm_q8_0_f32`, and `kernel_dsv4_hc_expand4` with F32 activations.
  The final 32 rows use the same 32-row legacy arithmetic tiles as the complete
  2,048-row projection, so no captured predecessor output tile is required.
- Captured full `attn_low`, `attn_out`, and `hc_attn_post` tensors from two
  fresh pinned DwarfStar processes over the canonical 2K prompt. Every pair
  was byte identical. Their full SHA-256 values are
  `edd7304f5f41313b19f432b4077c6bf08c97605c0bbaeb002e6afc749b768fa9`,
  `99d1251d729592a383a208258bf96579c5025fb35ed3215be26d0616f2094871`,
  and `19c0a248fce8b530bbc39f4c7e7ba0ff277b97466d8e2bc86ce199639b6739c1`.

Implementation:

- Added `prefill-attention-output-2048-v1` and a strict importer that requires
  both full repeated capture sets, their exact sizes, and their pinned hashes
  before retaining the final `32×8192`, `32×4096`, and `32×16384` tiles.
- Evolved `prefill-layer0-boundary-probe` to schema v4. Its one Rust-owned
  command buffer now executes 22 compute dispatches from token IDs through
  `hc_attn_post`. The two output weights are additional mmap-backed no-copy
  views, bringing the boundary to 13/13 pointer matches.
- Imported DwarfStar's exact expert-map/grouped-matmul schedule from the shared
  Metal source, retained the native dense Q8 projection, and applied the split
  post/combination weights directly to the live residual HC buffer. The five
  joined fixtures require 5,521,408 produced FP32 values to match by bit
  pattern, in addition to guarded cache checks and the reconstructed KV seam.

Target-Mac evidence:

- The focused optimized live run was C0 exact with 22 dispatches and 13/13
  no-copy model views, reporting 63.477 ms wall and 5.787 ms GPU. Setup,
  synchronization, exhaustive readback, and comparison remain in the wall
  interval; this is correctness evidence, not throughput.
- The complete target-Mac gate passed formatting, all 75 Rust tests, 51 Python
  tests, all 235 differential manifests, optimized Objective-C/Metal
  compilation, strict validation of all 1,288 target tensors, every established
  Metal/decoder control, and both steady-state benchmarks. Its fresh v4 run
  reported 65.043 ms wall and 7.116 ms GPU with 13/13 no-copy views. The long
  2K sequential diagnostic remained exact against its decode-replay oracle at
  19.138 tokens/s over 107012.259 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.

Decision and next:

- Accept the grouped Q8 attention output and HC post-update as the exact next
  segment of the native M1 final tile. Extend the same command buffer through
  the layer-0 FFN ingress and batched MoE/HC tail next; complete 2K native state
  construction and sparse ratio-4 indexed attention remain required before
  paired claims.

### 2026-08-15 — Continuous native M1 final tile crosses zero-prefix attention

Objective:

- Extend the exact final 32-row token-to-KV boundary through DwarfStar's active
  zero-prefix batched FlashAttention read and inverse-RoPE output.

Oracle evidence and schedule decision:

- Traced layer 0's `ratio == 0` batch path. DwarfStar attends over the complete
  contiguous 2,048-row `KVcur` tensor, uses a 128-token causal window, executes
  the non-vector block-map plus `kernel_flash_attn_ext_f16_dk512_dv512`, and
  then applies inverse tail RoPE to 64×512 output heads.
- Captured full `kqv_out` and `kqv_back` tensors from two fresh pinned
  DwarfStar processes over the canonical 2K prompt. Both pairs were byte
  identical. Their full SHA-256 values are
  `0678586b3fa811f40d053b4cef4f40c9172bb497d0ac1862dc43de7f5b04a1d9`
  and `79ec161cba188d6e9d1e0b3a84e3071e6c0cba53b5b7a56406d9ac6cf8ddabcc`.
- Kept continuity without replaying the first 2,016 rows: the runtime loads the
  repeated captured KV prefix as input, overwrites rows 2016--2047 with its
  live post-FP8 tile, and evaluates the rectangular 32-query by 2,048-key
  problem. The production range helper is intended for this geometry; the
  target-Mac C0 gate proves it matches the final rows of the square oracle.

Implementation:

- Added `prefill-attention-read-2048-v1` and its reproducible strict importer.
  The importer requires both complete capture pairs and their pinned hashes
  before retaining the 2,016×512 KV prefix and final 32×64×512 `kqv_out` and
  `kqv_back` tiles.
- Evolved `prefill-layer0-boundary-probe` to schema v3. One command buffer now
  executes 18 compute dispatches continuously from token IDs through the full
  final-tile attention result. It adds the F32-to-F16 full-KV stage, 8×64
  block-map specialization, eight-simdgroup non-vector attention kernel, and
  inverse RoPE. The sink weights are an eleventh mmap-backed no-copy model
  view.
- The four joined fixtures require 4,603,904 retained produced FP32 values to
  match by bit pattern. The probe also checks the complete reconstructed KV
  tensor, so captured prefix rows and the live replacement seam cannot drift
  silently.

Target-Mac evidence:

- The optimized live run was C0 exact across every prior and new boundary,
  preserved 11/11 mmap pointer identities, and reported 59.825 ms wall and
  5.249 ms GPU. Setup, synchronization, exhaustive readback, and comparison
  remain in the wall interval; this is correctness evidence, not throughput.
- The focused host suite passed 74 Rust tests, optimized Objective-C/Metal
  compilation, and strict validation of the new differential fixture.
- The complete target-Mac gate passed formatting, all 74 Rust tests, 50 Python
  tests, all 234 differential manifests, optimized Objective-C/Metal
  compilation, strict target-model inspection, every established
  Metal/decoder control, and both steady-state benchmarks. Its fresh v3 run
  reported 59.465 ms wall and 4.867 ms GPU with 11/11 no-copy views. The long
  2K sequential diagnostic remained exact against its decode-replay oracle at
  19.484 tokens/s over 105110.880 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.

Decision and next:

- Accept the rectangular zero-prefix attention read plus inverse RoPE as the
  exact continuation of the native M1 final tile. Next extend this same batch
  command buffer through the grouped Q8 attention output and HC post-update;
  complete 2K native state construction remains required before paired claims.

### 2026-08-15 — Continuous native M1 final tile reaches guarded KV storage

Objective:

- Extend the exact token-ID-to-Qcur final tile through the pinned M1 batch KV
  finalization path without claiming a complete 2K cache construction.

Oracle evidence and schedule decision:

- Traced the active batch path as standalone KV tail RoPE, in-place E4M3FN
  simulation for the 448 non-rotary channels, then F32-to-F16-to-F32 rounding
  and raw-ring scatter. It does not use the fused one-row decode KV-store
  kernel. Zero-prefix batch attention consumes the contiguous batch KV tensor;
  the raw ring preserves state for later chunks and decode.
- Captured full layer-0 `KVrope` and `KVcur` tensors from two fresh pinned
  DwarfStar processes over the canonical 2K prompt. Each corresponding pair
  was byte-identical. The full-capture SHA-256 values are
  `9a9642491e6ae5018a5dc5012eac67884458d4439f58a8bedb071c82dc3aaeb7`
  and `bef8d14d805a482960cbf7315ad0efccf211516a27bd767d0884b81f3ad33893`.
- Retained positions 2016--2047 and derived the exact raw-cache payload by
  applying IEEE-754 binary16 round-to-nearest-even and expansion back to
  binary32. Those rows map contiguously to physical ring rows 96--127. Earlier
  rows remain guards because the full 2K scatter aliases ring destinations and
  is outside this final-tile checkpoint.

Implementation:

- Added the strict `prefill-kv-state-2048-v1` fixture and reproducible importer.
  The manifest binds both full captures, records four production kernels, and
  distinguishes captured values from the derived cache-storage artifact.
- Evolved `prefill-layer0-boundary-probe` to schema v2. One Rust-owned Metal
  command buffer now executes 14 dispatches continuously from the final 32
  token IDs through KV RoPE, FP8 simulation, and F16-rounded cache storage.
  It snapshots pre-RoPE KV for the prior control, returns `KVrope`, `KVcur`, and
  the full 128-row raw ring, and requires sentinel rows 0--95 to remain intact.
- Added the F16-to-F32 contiguous conversion pipeline, extended the C/Rust ABI,
  stable JSON, host fixture/report checks, complete-gate integration text, and
  runtime/project documentation. Across the three joined fixtures, 2,506,752
  retained produced FP32 values must match by bit pattern.

Target-Mac evidence:

- The focused optimized live run was C0 exact across every old and new boundary,
  retained 10/10 mmap pointer identities, and proved all 49,152 guard-tile
  elements unchanged. The complete-gate rerun reported 57.703 ms wall and
  3.569 ms GPU for the cold 14-dispatch tile; setup, synchronization, exhaustive
  readback, and guard verification remain in scope.
- The complete gate passed formatting, 72 Rust tests, 49 Python tests, all 233
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  long 2K sequential diagnostic remained exact against its decode-replay oracle
  at 19.521 tokens/s over 104914.162 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.
- A final fixture-shape regression test raised the Rust suite to 73 tests; that
  complete host suite and `git diff --check` passed after the documentation
  update.

Decision and next:

- Accept token IDs through final-tile guarded KV storage as one continuous
  native M1 boundary. Next capture and implement the zero-prefix batch
  FlashAttention read/output for this same tile before broadening retained rows.

### 2026-08-15 — Continuous native M1 final tile now starts at token IDs

Objective:

- Remove the captured normalized-activation seam from the native layer-0 batch
  boundary while preserving DwarfStar's actual M1 schedule and exactness.

Oracle evidence and schedule decision:

- Traced the pinned M1 path from token embedding through `kernel_get_rows_f16`,
  four-stream `kernel_repeat_f32`, plain `kernel_rms_norm_f32_4`, the legacy
  `kernel_mul_mm_f16_f32` batch mixer, and fused HC split/collapse/learned norm.
  The M1 does not select the newer fused TensorOps projection for this path.
- Captured full 2K `hc_attn_pre` and `attn_norm` artifacts in two fresh
  DwarfStar processes. Corresponding hashes were byte-identical, and the
  attention-normalized hash also exactly matched the independently captured
  Q/KV fixture input.
- Retained only the final 32 token IDs and complete final-tile HC/norm outputs,
  while recording the full-capture hashes in a strict differential fixture.

Implementation:

- Added `prefill-layer0-boundary-probe` and schema
  `rust-star-prefill-layer0-boundary-probe-v1`. One command buffer now executes
  ten dispatches continuously from token IDs through Q head RMSNorm/RoPE.
- Wrapped ten GGUF tensor spans as mmap-backed no-copy Metal views, retained a
  pre-RoPE Q snapshot with an in-command-buffer blit, and required the HC and
  Q/KV fixtures' 2,457,600 produced FP32 values to match by bit pattern.
- Added the reproducible fixture importer, differential manifest, stable JSON,
  CLI and complete-gate integration, host tests, and runtime documentation.
  The report explicitly rejects full-layer, full-prefill, and throughput claims.

Target-Mac evidence:

- The focused live run retained 10/10 mmap pointer identities and matched every
  boundary bit-for-bit. It reported 56.433 ms wall and 2.544 ms GPU for the
  cold ten-dispatch tile; setup, synchronization, and exhaustive readback remain
  in scope.
- The complete gate passed formatting, 72 Rust tests, 49 Python tests, all 232
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  new boundary repeated C0 exact at 55.079 ms wall / 2.561 ms GPU. The long 2K
  sequential control remained exact against its decode-replay oracle at 19.579
  tokens/s over 104602.439 ms and preserved the expected native-batch mismatch.

Decision and next:

- Accept token IDs through Qcur as one continuous native M1 final-tile boundary.
  Extend it forward through KV RoPE/storage and batched attention before
  broadening row and layer coverage toward a complete eligible prefill path.

### 2026-08-15 — Native M1 batch boundary extended through layer-0 Q/KV setup

Objective:

- Extend the first exact batch projection through the actual layer-0 M1
  Q/KV schedule without substituting a debug-only or newer-GPU path.

Oracle evidence and schedule decision:

- Captured `q_lora`, `q_lora_norm`, `KVraw`, `KVnorm`, and `Qcur` across the
  canonical 2K prompt in two fresh DwarfStar processes. Every corresponding
  payload was byte-identical; the repeated `q_lora` hash also exactly matched
  the earlier boundary capture.
- Captured `Qraw` twice in two additional fresh processes. Both 256 MiB files
  were byte-identical. The hook selects the explicit Q-B path, but this does
  not replace active production arithmetic on the pinned M1 build: its fused
  Q-B/F16 entry point is unavailable and the normal schedule falls back to the
  same legacy Q8 batch projection followed by fused head RMSNorm/RoPE.
- Retained only positions 2016 through 2047, one complete legacy 32-row tile,
  while binding all seven full 2K capture hashes in a reproducible fixture.
  The seven payloads total 9,306,112 bytes.

Implementation:

- Added `prefill-qkv-boundary-probe` and schema
  `rust-star-prefill-qkv-boundary-probe-v1`. One Rust-owned Metal command wraps
  five GGUF ranges without copying and dispatches Q-A, KV-A, fused Q/KV learned
  RMSNorm, Q-B, and Q head RMSNorm/RoPE with DwarfStar's exact row/position
  geometry.
- Added a strict importer, differential manifest, stable privacy-limited JSON,
  CLI and complete-gate integration, host fixture/report tests, and runtime
  ownership documentation. The report explicitly rejects a full-prefill or
  throughput interpretation.

Target-Mac evidence:

- The first focused live run retained 5/5 mmap pointer identities and matched
  all 2,195,456 produced FP32 values bit-for-bit. It reported 11.200 ms wall and
  2.346 ms GPU for the isolated five-dispatch tile.
- The complete gate passed formatting, 70 Rust tests, 48 Python tests, all 231
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  new boundary repeated C0 exact at 12.023 ms wall / 2.817 ms GPU. The long 2K
  sequential control remained exact against its decode-replay oracle at 19.336
  tokens/s over 105913.888 ms and preserved the expected native-batch mismatch.

Decision and next:

- Accept the complete layer-0 Q/KV setup and Q-head boundary as native M1
  arithmetic. Move the input boundary backward through batch HC mixing and
  learned attention norm toward token IDs; in parallel sequence, extend this
  output boundary through KV RoPE/storage and batched attention.

### 2026-08-15 — First native M1 batched-prefill arithmetic boundary isolated

Objective:

- Trace the actual DwarfStar 2K prefill schedule on the M1 Ultra and implement
  the first exact Rust-owned batch kernel boundary without claiming a complete
  prefill path.

Oracle evidence and corrected hypothesis:

- Two fresh 2K graph captures were byte-identical for both layer-0
  `attn_norm` and `q_lora`. The final `attn_norm` row exactly equals the same
  row produced by 2,048 sequential one-token evaluations, proving the schedule
  divergence begins after normalization.
- All 1,024 final-row `q_lora` values differ between schedules; maximum absolute
  error is `3.62396240234375e-05`.
- A live diagnostic run printed that Metal 4 tensor kernels are disabled on the
  M1 Ultra. The earlier source-only assumption that this machine selected the
  retained N128 TensorOps kernel was false. The actual path is the legacy
  `kernel_mul_mm_q8_0_f32` with 32-row tiles, 128 threads, and 6,144 bytes of
  threadgroup memory. A repeated live dump matched the original batch capture
  SHA-256 exactly.

Implementation:

- Added a reproducible importer and compact fixture containing the final 128
  normalized rows, their native batch outputs, and the sequential final-row
  control. The three payloads total 2,625,536 bytes and bind two independent
  full-batch captures.
- Added a narrow C ABI and Rust command that wraps
  `blk.0.attn_q_a.weight` from the shared model mmap, specializes DwarfStar's
  batch kernel with both bounds constants false, dispatches four by sixteen
  groups, then runs the existing four-simdgroup one-row kernel over the shared
  final input.
- The live M1 probe matched all 131,072 batch outputs and all 1,024 sequential
  control outputs bit-for-bit. It reported 3.145 ms wall / 1.050 ms GPU for the
  isolated 128-row batch dispatch and retained the expected 1,024/1,024
  cross-schedule mismatch. These are boundary timings, not prefill throughput.
- The initial standalone TensorOps attempt was preserved as a failed
  hypothesis during development: it first failed because the conditional
  kernel was absent, then produced different arithmetic when force-enabled.
  The live DwarfStar device log resolved the discrepancy and the implementation
  was corrected to the native M1 path.

Validation:

- The complete target-Mac gate passed: formatting, 68 Rust tests, optimized
  Objective-C/Rust compilation, 47 Python tests, all 230 differential
  manifests, strict model inspection, and every retained live Metal control.
- The new boundary repeated C0 exact at 2.490 ms wall / 0.410 ms GPU. The long
  2K sequential control again matched its decode-replay oracle at 18.478
  tokens/s over 110832.479 ms and retained the expected 129,280-logit native
  batch mismatch.

Next:

- Complete documentation and the full target-Mac runtime gate, publish this
  checkpoint, then extend the batch path through layer-0 Q/KV setup.

### 2026-08-15 — 2K state ownership reached the batched-prefill boundary

Objective:

- Generalize cold state past 128 positions, reach the first protocol context,
  and determine whether sequential exact decode can initialize eligible
  benchmark state without importing DwarfStar's batched prefill path.

Oracle evidence:

- Captured the canonical first 2,048 token IDs and two fresh native DwarfStar
  2K frontier-logit files. Both batch captures were byte-identical after FP32
  reconstruction, selected token 15342, and have payload SHA-256
  `7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf`.
- Added a narrow DwarfStar decode-replay capture helper: one-token cold prefill
  followed by 2,047 ordinary `ds4_session_eval` calls over the same canonical
  IDs. Two fresh replay logits were byte-identical, selected token 15342, and
  have SHA-256
  `aa657efb7a5cb7108ee639ea797eb4f1c223f36360d356f96674499935d1f405`.
- The batched and replay tensors differ at all 129,280 logits with maximum
  absolute error 2.325326. Preserve this as an arithmetic-boundary result, not
  a reason to weaken C0 or relabel sequential decode as prefill.

Implementation:

- Added an immutable context-capacity field to the Rust/Objective-C layer ABI.
  Raw KV now uses a 128-row physical ring with ordered wrapped-window staging;
  compressed capacity is derived independently from each layer's ratio.
- Corrected FlashAttention padding from context-scaled allocation to the two
  32-row tiles actually consumed by the vector kernel. Added an explicit guard
  after 512 ratio-4 compressed rows until sparse indexer selection exists.
- Kept synchronized multi-layer control scratch at the uniform ratio-4 maximum
  so those legacy controls can safely reuse unscoped buffers across the layer-2
  transition while compressed-cache ownership remains ratio-specific.
- Added `prefill-frontier-probe`, a three-payload differential fixture covering
  canonical token IDs, native batch logits, and decode-replay logits, stable
  diagnostic JSON, CLI/runtime-gate coverage, and fixture/report tests.

Target-Mac evidence and decision:

- The unchanged one-token/position-127 cold control remained C0 exact and
  reported 18.533 evaluated decode positions/s after the ring rewrite.
- The 2K sequential run reported 18.603 tokens/s over 110090.070 ms, matched
  every decode-replay logit bit, and selected token 15342. It explicitly
  reports batched-prefill C0 false and is not an engine measurement.
- The final complete gate passed 66 Rust tests, 46 Python tests, all 229
  differential manifests, optimized Objective-C/Metal compilation, strict
  target inspection, and every retained live Metal control. Its cold command
  reported 843.365 ms for one-token initialization and 17.670 evaluated decode
  positions/s. Its independent 2K sequential diagnostic again matched every
  replay-logit bit at 18.252 tokens/s over 112206.922 ms and preserved the
  expected 129,280-logit batched-prefill mismatch.
- Accept context-sized state ownership and the raw-ring transition as the next
  exact decode boundary. Reject the hypothesis that sequential decode can stand
  in for native batched prefill under C0. Import batched prefill next, then add
  sparse indexed decode and the measurement producer.

### 2026-08-15 — Captured initial state replaced by exact cold prefill

Objective:

- Construct position-0 raw KV and compressor state from the live prompt token
  rather than copied oracle rows, while retaining the complete position-127
  transcript and cache-emission gates.

Oracle evidence:

- Verified that the first 26-space prefix of the pinned benchmark input
  tokenizes to the single token 36662. Two fresh pinned DwarfStar processes
  wrote the full logits immediately after raw one-token prefill; both 517,120
  byte payloads were identical, selected token 201, and have SHA-256
  `a4973c1e1f53bf1659a9a15e66c3186d03432810e7606387c55f8ee083ffba35`.
- Added a strict prefill decode-step fixture and reproducible two-capture
  importer. The differential registry now contains 228 manifests.

Implementation:

- Added an explicit cold initial-state mode to the stable Rust/Objective-C
  layer boundary. Position 0 clears raw caches, recurrent compressor state, and
  persistent compressed rows, then the ordinary layer path writes raw KV row 0
  and seeds each compressor from its live attention-normalization activation.
- Position 1 in cold mode continues that state and cannot execute the captured
  cache-row/compressor-prime initialization branch. Existing commands retain
  the captured mode as independent regression controls.
- Added `cold-prefill-decoder-probe`, stable diagnostic JSON, CLI help, complete
  gate coverage, host fixture/report tests, and ownership documentation. It
  compares every position-0 logit before running the unchanged 127-position
  feedback loop and final ratio-128 checks.

Target-Mac evidence and decision:

- The first live cold-state process matched all prefill logits, all 128
  committed tokens, final token 33148, every final logit, and both first live
  ratio-128 rows. It reported 3379.018 ms prefill and 0.729 evaluated
  positions/s while paying cold model-residency costs.
- An immediate fresh-process repeat remained bit-identical and reported
  1130.755 ms prefill plus 18.246 evaluated positions/s with resident weights.
  Preserve both as diagnostics; their residency conditions differ and the
  one-token prompt is not a paired-protocol frontier.
- The complete gate passed 64 Rust tests, 45 Python tests, all 228 differential
  manifests, optimized Objective-C/Metal compilation, strict model inspection,
  every retained live Metal control, and both position-127 commands. Its cold
  command reported 1078.665 ms prefill and 18.311 evaluated positions/s.
- Accept removal of captured initial state at the one-token frontier. The next
  checkpoint is multi-token prefill with context-sized raw/compressed ownership
  and arbitrary-frontier initialization, followed by the measurement producer.

### 2026-08-15 — Integrated decoder crossed position 127 exactly

Objective:

- Extend the exact greedy decoder from its four-position control through the
  complete 128-token oracle transcript and verify the first ratio-128
  compressor emissions in live end-to-end execution.

Oracle evidence:

- Ran the pinned DwarfStar executable twice in fresh processes for 127
  evaluated positions, retaining initial committed token 201 plus all 127
  selected tokens and the final full-vocabulary logits. Both captures were
  byte-identical. The token payload SHA-256 is
  `95aedd05c1843ed9638bcd24e93b9ed4cde360a341f8c720c7efc908c3586697`;
  the position-127 logits SHA-256 is
  `1258d4c3ab662f72ac1f70be80ddbc8f9cf72db35b4450231369f3b6ae07c895`.
- Imported one strict decode-step fixture containing 128 token IDs and all
  129,280 final logits. The final selected token is 33148, and the differential
  registry now contains 227 manifests.

Implementation:

- Added `position127-decoder-probe`, its stable diagnostic JSON schema, CLI and
  complete-gate coverage. One prepared executor advances positions 1–127,
  feeds each lowest-ID argmax into the following step, and retains all
  layer-scoped raw and compressed state.
- Added a synchronization-only, layer-scoped compressed-cache row readback.
  The final C0 phase compares the complete token transcript, final logits, and
  live layer-3/layer-5 ratio-128 rows. Its timer ends before those readbacks and
  comparisons.
- Added a reproducible frontier-fixture importer plus Rust fixture/JSON and
  Python manifest tests.

Failed hypotheses and correctness fixes:

- The first integrated run diverged at committed token 16. The decoder used a
  fixed compressed-RoPE origin; the correct origin is the beginning of the
  completed compression window, `position + 1 - ratio`.
- After token exactness was restored, the live ratio-128 row was zero. A
  temporary synchronized work-buffer check localized the failure before the
  persistent-cache blit: packed KV, score, and softmax scratch allocations had
  room for eight rows instead of the active 128-row window. All compressor
  scratch is now sized from its ratio. The temporary diagnostic was removed.

Validation and decision:

- The live M1 Ultra command reproduced all 128 tokens, final token 33148, every
  final logit, and both first ratio-128 rows bit-for-bit. The complete gate
  reported 16.493 evaluated positions/s over 127 positions and passed 62 Rust
  tests, 44 Python tests, all 227 differential manifests, the optimized
  Objective-C/Metal build, strict model inspection, and every retained live
  Metal control.
- Accept the position-127 integrated frontier as C0 exact, but not as paired
  throughput. It begins from captured one-token cache/compressor state. The
  next checkpoint is cold prefill plus arbitrary-frontier initialization,
  followed by the engine-measurement producer.

### 2026-08-15 — Persistent compressed memory reused exactly at position 4

Objective:

- Remove the captured position-3 ceiling at the first semantically meaningful
  boundary: execute the next sampled token while retaining and consuming the
  ratio-4 compressed rows emitted by the preceding step.

Oracle evidence:

- Ran the preserved pinned DwarfStar executable twice in fresh processes with
  one prefill token and four generated tokens, batch-dumping every retained
  position-4 boundary for layers 0–42. All 1,376 payload pairs were
  byte-identical; their ordered aggregate SHA-256 is
  `73f0af1597da79b5e53c19716822c105df6256e353ee18d8c013d4809c74c3cb`.
- Captured the five terminal output-head tensors twice in two additional fresh
  processes. All pairs were byte-identical with aggregate SHA-256
  `8ca919fe0b7cc11ff861b9de6f576ba0f736b1fb4376878ff4d464cabc594054`.
  Position 4 consumes token 262 and lowest-ID argmax selects token 1554.
- Imported 43 complete layer fixtures plus one output-head fixture. The strict
  differential registry now contains 226 manifests.

Implementation:

- Replaced the four-row raw-cache allocation with a 128-row frontier and added
  32 persistent compressed attention/indexer rows per compressed layer. The
  bounded executor accepts positions 1–127, which covers the protocol's future
  128-token decode without yet claiming that full execution path.
- Preserved emission-step ordering: a new compressed row remains in its work
  buffer for the attention encoder that produced it, then a Metal blit commits
  it to persistent cache storage. Later positions stage all committed rows;
  emission positions stage prior committed rows followed by the new work row.
- Extended both explicit-input and feedback-loop C0 paths through position 4,
  updated their schemas to v2, and retained the earlier three-position
  transformer-only commands as independent controls.

Validation:

- The live M1 Ultra closed loop is C0 exact for all 172 layer/position
  boundaries and all 517,120 logits. It generated
  `361 -> 1915 -> 262 -> 1554`; the complete gate reported 18.798 tok/s
  overall and 18.916 tok/s steady over the diagnostic timed path.
- The timing interval still includes transformer submission/execution, output
  execution, logits transfer, and CPU argmax while excluding correctness
  collection. It remains paired-protocol ineligible because cold prefill and
  arbitrary-frontier initialization are not implemented.
- The complete gate passed 60 Rust tests, the optimized macOS
  Objective-C/Metal build, 43 Python tests, all 226 fixture manifests, strict
  real-model inspection, and every established Metal control.

Decision:

- Accept persistent compressed-cache lifetime and first reuse as exact. The
  next checkpoint is a 128-step generated-token run through the already-sized
  frontier, followed by cold prefill and engine-measurement output.

### 2026-08-15 — Three-position decoder loop closed and timed honestly

Objective:

- Feed exact greedy selections back into the following decoder positions and
  separate exhaustive C0 tensor collection from the measured execution path.

Implementation:

- Added `closed-loop-decoder-probe` and schema
  `rust-star-closed-loop-decoder-diagnostic-v1` without changing the existing
  explicit-input `decoder-output-probe` control.
- The exhaustive pass starts with token 201, derives 361, feeds 361 into
  position 2, derives and feeds 1915 into position 3, and finally selects 262.
  Every retained boundary across 43 layers plus all 129,280 logits per position
  remains C0 bit-identical.
- Added a prepared output-head mode that omits the four intermediate host
  copies while retaining the full logits copy required for CPU greedy argmax.
  The timed pass does not invoke any transformer collector or fixture
  comparison. Pipeline preparation and the exhaustive C0 allocation pass occur
  before its step intervals.
- Stable JSON explicitly classifies the result as diagnostic, records whether
  correctness/logits/argmax occur in the interval, and marks paired-protocol
  eligibility false with the remaining cold-prefill/arbitrary-frontier blocker.
- Added CLI help, runtime-gate coverage, host JSON tests, and runtime/benchmark
  documentation.

Validation and measurement:

- Formatting, 60 Rust tests, the optimized macOS Rust/Objective-C/Metal build,
  JSON parsing, and diff checks pass.
- The complete target-Mac gate passed formatting, 60 Rust tests, 42 Python
  tests, optimized compilation, all 182 fixture envelopes, every retained live
  Metal control, and the new closed-loop command.
- The live model run selected `361, 1915, 262` in both exhaustive and timed
  loops. The full-gate timed steps were 54.610, 51.727, and 53.989 ms, giving
  18.712 complete tok/s and 18.919 steady tok/s over the final two tokens.
- This is not a headline or paired result: it begins from the captured
  position-0 cache/compressor state, uses only three positions, and performs no
  cold prompt prefill.

Next:

- Generalize state allocation and execution to cold prefill and positions past
  three, then drive 128 closed-loop tokens into the engine-measurement contract.

### 2026-08-15 — Exact decoder output boundary completed

Objective:

- Extend the exact 43-layer transformer stack through DwarfStar's output HC
  collapse, learned output normalization, full vocabulary logits, and
  deterministic next-token selection for positions 1–3.

Oracle evidence:

- Captured `result_hc_pre`, `result_hc_weights`, `result_hc`, `result_norm`,
  and `result_output` for terminal decode positions 1, 2, and 3 in two fresh
  pinned DwarfStar processes. All 15 corresponding payload pairs were
  byte-identical; their ordered aggregate SHA-256 is
  `3cda1544443ba220b5019201c0232704cc0f177b8ab644689c09d24e70d40728`.
- Imported three strict output-head fixture envelopes containing 549,920 bytes
  each. Full-logit lowest-ID argmax selects token 361 at position 1, token 1915
  at position 2, and token 262 at position 3.

Implementation:

- Added `decoder-output-probe` and schema
  `rust-star-decoder-output-position-advancing-probe-v1`.
- Added a no-copy Metal output boundary that consumes the retained layer-42 HC
  buffer, wraps five output model ranges directly from the GGUF mmap, and
  dispatches the plain HC norm, F16 HC projection, four-value HC weighting,
  fused HC collapse/learned norm, and 129,280-row Q8_0 vocabulary projection.
- Preserved the existing `layers0-42-decode-probe` as an independent
  transformer-stack control. The new correctness schedule explicitly reports
  44 command buffers and two host waits per step.
- Added stable JSON writing, CLI/runtime-gate integration, output-fixture shape
  and argmax tests, documentation, and a repeated-capture fixture importer.
- The first live attempt differed by one bit at logit 0 while all preceding
  output tensors were exact. Tracing DwarfStar's dispatch selection showed that
  output dimensions above 65,536 force eight simdgroups; using the ordinary
  four-simdgroup projection changes the reduction order. A dedicated
  eight-simdgroup output pipeline restored C0 without changing earlier Q8
  projection controls.

Validation:

- The complete target-Mac gate passed formatting, 59 Rust tests, 42 Python
  tests, optimized macOS compilation, the cross-language artifact contract,
  strict validation of all 182 differential fixtures, every prior live Metal
  control, and the new three-position decoder-output probe.
- All five output tensors were bit-identical at all three positions, including
  every FP32 bit in each 129,280-value logit vector.
- In the final correctness gate the output-head command reported 15.025/2.946
  ms wall/GPU at cold position 1, 1.679/1.355 ms at position 2, and 1.691/1.359
  ms at position 3. These intervals include correctness-oriented scheduling
  and readback and are not token-throughput measurements.
- Generated JSON evidence remains under the ignored
  `rust-star/.work/runtime-target/` directory. The user's two untracked
  console logs remain untouched.

Decision:

- This is an exact explicit-input decoder output boundary, not yet a
  closed-loop generator. The selected token is computed and reported, but the
  next input remains supplied by the fixed regression sequence. No committed
  token-throughput claim is eligible yet.

Next:

- Feed selected tokens directly into subsequent steps, then separate exhaustive
  C0 collection from the timed path and emit the Rust Star
  engine-measurement contract for paired benchmarking.


### 2026-08-15 — Ordered exact execution extended through all 43 layers

Objective:

- Complete the transformer-stack boundary through layer 42 while retaining the
  published four-, six-, and eight-layer controls and every C0 contract.

Oracle evidence:

- Batch-captured positions 0–3 for layers 8–42 twice in fresh pinned
  DwarfStar processes. All 3,448 corresponding payload pairs, totaling
  78,356,784 bytes per run, were byte-identical. Their ordered aggregate
  SHA-256 is
  `0c63afebdf0b0e40ed48f2efb2c06b3b7fe193e420d7456de2a8fdbf84cd1f29`.
- All eight capture CSVs recorded one context/prefill token and three generated
  tokens. Position 3 contained `KVcompress` rows for exactly the 18 even layers
  from 8 through 42 and none of the odd layers.
- Imported 140 new strict fixture envelopes: one position-0 compressor prime
  and complete position-1/2/3 boundaries for each remaining layer. The
  repository now contains 179 validated differential-fixture manifests.

Implementation:

- Replaced layer-4–7 pair-specific complete-fixture, cache-row, compressor
  prime, and compressed-row branches with a checked registry covering layers
  4–42. The registry has explicit bounds and a separate even-layer compressed
  output table.
- Widened the checked contiguous scheduler and Objective-C layer/tail guards to
  43 layers without changing compressor arithmetic or the existing control
  commands.
- Added `layers0-42-decode-probe`, schema
  `rust-star-layers0-42-position-advancing-probe-v1`, atomic JSON output,
  runtime-gate coverage, registry/shape tests, and a reproducible repeated-run
  fixture importer.

Validation:

- The complete target-Mac gate passed: formatting, 57 Rust tests, optimized
  Objective-C/Metal build, 42 Python tests, all 179 fixture manifests, artifact
  interoperability, strict real-model inspection, every established no-copy,
  layer, scheduler, compressor, and benchmark control, plus the new 43-layer
  command.
- The 43-layer path was C0 exact for all 129 layer/position boundaries. The
  final gate reported 837.045/800.215 ms at position 1, 62.799/61.229 ms at
  position 2, and 53.374/49.691 ms at position 3 (wall/summed GPU).
- It retains 43 raw KV caches, validates every position-3 even-layer compressed
  KV row through layer 42, and exposes layer 42's exact 16,384-element HC state.
  Setup, synchronization, and exhaustive comparison remain in scope, so these
  timings are correctness diagnostics rather than token-throughput claims.

Decision:

- Accept the full transformer-stack boundary. The next exact gate is output
  normalization, logits, and token selection; do not call the current command
  a complete decoder until that boundary is implemented and verified.

### 2026-08-15 — Ordered exact execution extended through layer 7

Objective:

- Continue the live position-advancing HC handoff through the next ratio-4 and
  ratio-128 layer pair while retaining every established scheduler and
  compressor control.

Oracle evidence:

- Captured layer-6/7 position-0 primes and complete position-1/2/3 boundaries
  twice in fresh pinned DwarfStar processes. All 197 corresponding payload
  pairs were byte-identical.
- Layer 6's 512-value position-3 compressed KV row has SHA-256
  `63ec3c2a44f2eb9a5d73ba67d05d8699bde608cd43b98382465754bf78352a33`.
  Layer 7 correctly emitted no ratio-128 row at this frontier.
- Added eight strict fixture envelopes. The repository now contains 39
  validated differential-fixture manifests.

Implementation:

- Extended expected-tensor embedding, position-specific fixture selection,
  compressor priming, cache-row ownership, and compressed-emission comparison
  through layers 6 and 7.
- Widened the prepared contiguous-submission contract and Objective-C layer and
  command-tail guards from six to eight layers. The parity-derived compressor
  and biased-top-k paths required no arithmetic changes.
- Added `layers01234567-decode-probe`, stable JSON reporting, atomic output,
  CLI/runtime-gate coverage, host tests, fixture validation, and a reproducible
  importer backed by the existing repeated-capture construction.
- Preserved the four-layer and six-layer commands unchanged as independently
  executed controls.

Validation:

- The complete target-Mac gate passed: formatting, 55 Rust tests, optimized
  build, 42 Python tests, all 39 fixture manifests, artifact interoperability,
  strict real-model inspection, every no-copy and layer/scheduler probe, the
  four-, six-, and eight-layer position-advancing controls, the bounded
  position-127 replay, and both steady-state harnesses.
- The eight-layer path was C0 exact at every retained layer 0–7 boundary. The
  final gate reported 223.448/190.007 ms at position 1, 16.989/16.442 ms at
  position 2, and 17.782/17.160 ms at position 3 (wall/summed GPU).
- These intervals include correctness-oriented setup, synchronization, and
  exhaustive comparison and are not decoder-throughput claims.

Decision:

- Accept the eight-layer frontier. Before repeating pair-specific plumbing 17
  more times, batch-capture the remaining layers and introduce a checked layer
  fixture registry that can scale the same exact executor to layer 42.

### 2026-08-15 — First ratio-128 emissions matched at position 127

Objective:

- Cross the first layer-3 and layer-5 ratio-128 compressor emissions with a
  narrowly scoped, independently repeatable GPU replay, while making no token
  sampling or complete-decoder claim.

Oracle evidence:

- Captured positions 0 through 127 plus the position-127 `KVcompress` emission
  for layers 3 and 5 twice in fresh DwarfStar processes. All 258 corresponding
  activation and output payload pairs were byte-identical.
- The layer-3 emitted row has SHA-256
  `a0416831e464ab8652402b7fbcf854f1a1a401e878906379cc0382b1d1edcc30`;
  layer 5 has SHA-256
  `e63dc05a3efa88affe2e1483f88ea7a9212006730d368f193ff940abd66b230c`.
- Added two strict layer-segment fixtures, each containing 128 independently
  repeated 4,096-value `attn_norm` rows and one 512-value compressed output.
  The repository now has 31 validated differential-fixture manifests.

Implementation:

- Generalized the compressor pack/reduction kernels across ratio 4 and ratio
  128 while preserving the existing ratio-4 state-shift behavior.
- Added a standalone Rust-owned replay boundary that wraps the APE, KV, gate,
  and norm weights directly from the GGUF mmap, owns the 128-row activation and
  compressor state, encodes 263 dispatches in one command buffer per layer, and
  emits only at position 127.
- Added stable JSON reporting, a CLI command, complete-gate coverage, strict
  fixture tests, and a reproducible importer that rejects disagreement between
  repeated oracle captures.
- The report records the external activation boundary and explicitly sets
  sampling and full-decoder claims to false.

Validation:

- The complete target-Mac gate passed: formatting, 53 Rust tests, optimized
  build, 42 Python tests, all 31 fixture manifests, artifact interoperability,
  strict real-model inspection, every earlier no-copy/layer/scheduler probe,
  the new ratio-128 replay, and both steady-state harnesses.
- The existing four-layer and six-layer position-advancing controls remained C0
  exact through positions 1, 2, and 3 after the shared compressor path changed.
- Layer 3's 512-value position-127 output was C0 exact at 32.461 ms wall and
  8.236 ms GPU; layer 5 was C0 exact at 6.881 ms wall and 6.273 ms GPU. These
  include setup, synchronization, and comparison and are not throughput claims.

Decision:

- The first ratio-128 emissions are now correctness-gated. Preserve this replay
  as a bounded control and resume extending the ordered executor through the
  remaining decoder layers; do not describe the replay as token generation.

### 2026-08-15 — Compressor ownership generalized through layers 4–5

Objective:

- Replace the layer-2/3-specific compressor schedule with parity-derived
  ownership, extend the exact position-advancing executor through the next
  ratio-4/ratio-128 pair, and keep the published four-layer path as a control.

Oracle evidence:

- Captured all retained layer-4/5 boundaries at positions 0–3 twice in fresh
  DwarfStar processes. All 197 corresponding payloads were byte-identical.
- Layer 4 emitted its 512-value `KVcompress` row at position 3; both captures
  have SHA-256
  `30550322349818d5524c864ca624613c94f18feea0c2f9a57186afd9c8fff5a1`.
  Layer 5 correctly emitted no compressed row at this ratio-128 frontier.
- Added eight strict fixture envelopes: one position-0 compressor prime and
  complete position-1/2/3 boundaries for each layer. The repository now has 29
  validated differential-fixture manifests.

Implementation:

- Derived compressor ratio, indexer ownership, and emission cadence from layer
  parity for every compressed layer. Even layers use ratio 4 and own the
  indexer compressor; odd layers use ratio 128 and do not.
- Extended exact layer scheduling and the checked prepared-submission tail
  through layer 5. The common executor accepts a contiguous three-to-six-layer
  prefix, while the existing layers-0/3 entry point remains unchanged.
- Added the `layers012345-decode-probe` command, atomic JSON reporting, runtime
  gate coverage, and independently embedded layer-4/5 fixtures and primes.
- Added a reproducible importer that verifies both oracle captures before
  creating the canonical layer-4/5 fixture envelopes.

Validation:

- The complete target-Mac gate passed: formatting, 51 Rust tests, optimized
  build, 41 Python tests, all 29 differential fixtures, cross-language artifact
  checks, strict inspection of 1,288 required model spans, every incremental
  Metal probe, both position-advancing controls, and steady-state benches.
- The integrated four-layer control remained C0 exact at positions 1–3. The
  new six-layer path was also C0 exact at every retained boundary and reported
  143.285/108.618 ms, 14.098/13.178 ms, and 11.552/10.111 ms wall/summed GPU at
  positions 1, 2, and 3 respectively.
- These timings include correctness-oriented execution and are diagnostics,
  not decoder-throughput or end-to-end model-speed claims.

Decision and next gate:

- The next distinct compression boundary is position 127. Plan a bounded
  recurrent-state replay using the pinned oracle token sequence, beginning
  with layer 3 if necessary, and explicitly avoid presenting externally
  supplied tokens as sampling or a complete decoder.

### 2026-08-15 — First ratio-4 compressed KV emission matched at position 3

Objective:

- Cross the first compressed-attention boundary with real recurrent state,
  preserve the existing four-layer scheduler/cache controls, and require the
  emitted row plus all downstream boundaries to remain C0 exact.

Oracle evidence:

- Captured every retained layer 0–3 boundary at position 3 twice in independent
  fresh DwarfStar processes. All 131 corresponding payloads were byte-identical.
- Resolved the position-3 token as 1915 by uniquely matching the layer-0 hash
  router row. The standard selected routes were `[133,217,222,94,234,246]`,
  `[107,58,141,226,233,88]`, `[90,98,196,23,62,19]`, and
  `[64,87,198,214,128,1]` for layers 0 through 3.
- Captured layer 2's 512-value `KVcompress` row twice; both copies have SHA-256
  `73cbd1e2e062b9b52e67223b8fe024a7992eeaf9be49a461ebdf7d659ebd32c8`.
  Also captured independent layer-2/3 position-0 `attn_norm` inputs so Rust Star
  can prime the recurrent state without substituting a synthetic activation.
- Added four position-3 complete fixture envelopes and two position-0 prime
  envelopes. Layer 2's complete fixture contains 33 tensors and 741,808 bytes;
  the other complete fixtures contain 32 tensors and 739,760 bytes each.

Schedule correction:

- The earlier journal/README statement that layers 2 and 3 both used ratio 4
  was incorrect. DwarfStar's schedule alternates ratio 4 on even compressed
  layers and ratio 128 on odd compressed layers. Therefore layer 2 emits both
  attention and indexer compressed rows at position 3; layer 3 only updates its
  attention state and will first emit at position 127. Historical entries are
  retained unchanged; this entry and Current State are authoritative.

Implementation:

- Added persistent layer-scoped attention-compressor state for layers 2/3 and
  indexer-compressor state for layer 2, including exact position-0 priming.
- Mirrored the pinned M1 Ultra path: paired F16 matvec with two rows per
  threadgroup, separate one-row APE state store, and the legacy single-emission
  concat/softmax/multiply/sum reduction. The newer M3/M5 fused store and direct
  ratio-4 pool paths are not used.
- Added learned weighted RMS norm, compressed RoPE, ratio-4 state shift, FP8 KV
  quantization, and indexer Hadamard/FP4 QAT. The emitted layer-2 KV row is
  staged after four raw rows and consumed by FlashAttention at position 3.
- Grew each raw cache to four rows and the mixed attention staging capacity to
  five rows. Layer 2 now binds 33 no-copy model ranges, layer 3 binds 29, and
  layers 0/1 remain at 25.

Target-Mac evidence:

- `layers0123-decode-probe` passed at positions 1, 2, and 3. The emitted layer-2
  compressed row matched all 512 FP32 bit patterns, and every retained
  attention, router, expert, HC, and raw-cache boundary remained exact through
  layer 3.
- The first passing run reported 110.896/77.239 ms at position 1,
  8.863/8.123 ms at position 2, and 10.455/9.618 ms at position 3
  (wall/summed GPU). These intervals include correctness-probe behavior and are
  not decoder-throughput measurements.
- The pre-documentation gate passed 48 Rust tests and the optimized macOS run.
  The completed repository gate passed formatting, 49 Rust tests, 40 Python
  tests, all 21 differential fixture envelopes, the optimized macOS build, the
  strict 1,288-tensor target recipe, and every earlier incremental,
  synchronized, chained, and steady-state Metal control. Its integrated
  position-advancing run reported 110.111/74.626 ms at position 1,
  7.014/6.520 ms at position 2, and 9.448/7.164 ms at position 3
  (wall/summed GPU), all C0 exact.
- The first full-gate attempt exposed a lifetime bug outside the chained path:
  synchronized layers 2 and 3 shared compressor scratch keys despite their
  different state sizes. Compressor and indexer allocations are now always
  keyed by layer identity. The failed control passed before the complete gate
  was rerun from the beginning.

Next:

- Generalize the compressor schedule/state plumbing for the next layer pair,
  then design a bounded position-127 oracle-token gate for layer 3's first
  ratio-128 emission.

### 2026-08-15 — Four-layer state advanced across two exact decode positions

Objective:

- Replace the fixed token-201/position-1 assumption with the smallest real
  position-advancing loop while preserving the proven four-layer executor and
  keeping compressed-cache behavior outside the claimed boundary.

Oracle fixture:

- Confirmed the first two generated token IDs as 201 and 361 and captured every
  retained layer 0–3 boundary at position 2 twice in separate fresh DwarfStar
  processes.
- Each layer fixture contains 32 tensors and 739,760 verified bytes. All paired
  payloads were byte-identical; layer 0 declares 30 operations and layers 1–3
  declare 28 each.

Implementation:

- Added explicit token and position propagation through Rust, the stable C ABI,
  and the Objective-C shim. RoPE, inverse RoPE, raw-cache target rows, visible
  attention geometry, and hash-router tokens now advance with the step.
- Added `layers0123-decode-probe` and schema
  `rust-star-layers0123-position-advancing-probe-v1`. One prepared executor
  submits four command buffers with one tail wait at each position and retains
  four independent three-row cache allocations across both steps.
- Strengthened cache validation to cover every retained row and the newly
  written row after DwarfStar's exact FP16 cache-store rounding. The declared
  output handoff is layer 3's final 16,384-element HC state.
- The first position-2 run exposed a stale launch assumption: the FP32→FP16
  attention staging dispatch covered only the old 1,024-element/two-row shape.
  Scaling its threadgroups to the visible element count covered all 1,536
  elements and restored exact attention output.

Target-Mac evidence:

- Position 1/token 201 grew every cache to two rows and matched all four pinned
  layer fixtures. Position 2/token 361 preserved that history, grew every cache
  to three rows, and matched all four new fixtures at every retained boundary.
- The first successful run reported 103.106 ms wall / 70.862 ms summed GPU for
  the cold position-1 step and 7.333 ms wall / 6.347 ms summed GPU for position
  2. These values include probe correctness work and are not token throughput.
- The complete host/fixture portion passed 48 Rust tests, 38 Python tests, all
  15 differential fixtures, the optimized macOS build, and the strict
  1,288-tensor target recipe. After an output-poll cancellation, validation
  resumed from the first unexecuted Metal stage: every remaining incremental,
  full-layer, synchronized, chained, stateful, and steady-state control passed.
  The integrated stateful run reported 99.392/69.763 ms wall/summed-GPU at the
  cold first step and 8.943/6.455 ms at position 2. The 20-sample four-layer
  replay and 30-sample layer-0 control also completed with exact final readback.

Decision:

- Accept positions 1→2 as the minimum stateful decoder slice. Do not call it a
  complete decoder: 39 layers, the output head, logits, and sampling remain, and
  compressed cache state has not yet been emitted.

Next:

- Capture and implement the position-3 compressor/indexer boundary for layers
  2–3, where compression ratio four emits its first compressed-cache row.

### 2026-08-15 — Prepared four-layer replay isolated steady-state execution

Objective:

- Remove fixture decoding, model-directory lookup, host allocation, and tensor
  readback from the repeated four-layer timing interval while retaining one
  exhaustive post-measurement C0 gate.

Implementation:

- Added `rust-star-layers0123-steady-state-v1` and the
  `layers0123-bench` CLI with bounded warmup/iteration controls and atomic
  JSON output.
- Introduced prepared layer executions that resolve and validate 25 mmap-backed
  model spans per layer, decode each pinned fixture, and allocate every host
  result vector once. Existing single and chained correctness probes now use
  the same preparation object.
- Added a timing-only Metal command mode. It requires a completed declared
  chain, may be queried only through layer 0, and returns the first-submit to
  tail-completion wall interval plus the sum of all four command-buffer GPU
  intervals. It copies no activation, router, expert, cache, or HC data.
- Warmup and measured chains reuse one context, cached model views, activation
  allocations, and per-layer KV storage. After the last measured sample, the
  standard collector reads all four layers once and performs the unchanged
  bitwise comparisons.

Target-Mac evidence:

- A 2-warmup/5-sample smoke test reported 6.709 ms wall median and 5.524 ms
  summed-GPU median, then matched every final layer boundary.
- The canonical 5-warmup/20-sample run reported wall samples
  `[5.375, 4.769, 5.769, 4.265, 4.380, 4.062, 4.601, 4.197, 4.240,
  4.166, 5.656, 4.371, 4.525, 4.562, 4.231, 4.255, 4.008, 4.831,
  4.181, 4.212]` ms: 4.318 ms median, 0.180 ms MAD, 4.008 ms min,
  and 5.769 ms max.
- Summed-GPU timing was 3.767 ms median, 0.131 ms MAD, 3.602 ms min,
  and 4.851 ms max. One exhaustive collection after sample 20 matched all four
  pinned DwarfStar fixtures and expected expert routes bit-for-bit.
- The complete repository gate then passed 45 Rust tests, the optimized build,
  37 Python tests, all 11 differential fixtures, the strict 1,288-tensor model
  recipe, every incremental Metal probe, both four-layer scheduler controls,
  the new 5-warmup/20-sample replay, and the 30-sample layer-0 benchmark. Its
  integrated replay measured 4.241 ms wall median (0.139 ms MAD) and 3.781 ms
  summed-GPU median (0.110 ms MAD), followed by an exact final four-layer C0
  collection. The subsequent layer-0 control measured 1.900 ms wall median
  and 1.526 ms GPU median.

Decision:

- Accept prepared four-layer replay as the current steady-state
  execution/scheduler microbenchmark. Do not compare its fixed token-201,
  position-1 replay directly with DwarfStar token throughput.

Next:

- Keep this replay in the target-Mac gate, then either cross the next distinct
  layer boundary or introduce the minimum position-advancing decoder loop.

### 2026-08-15 — Layer 3 crossed the biased-top-k boundary and stayed exact

Objective:

- Extend both scheduler controls through layer 3, the first layer after the
  model's three token-hash router layers, and remeasure command chaining over
  the larger exact slice.

Oracle fixture:

- Captured all 32 retained layer-3 position-1 boundaries twice from separate
  fresh processes using the pinned DwarfStar executable; every corresponding
  artifact was byte-identical. Captured position-0 `KVcur` independently and
  derived the FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer3-complete-v1/`: 28 ordered operations and
  33 tensors totaling 741,808 verified bytes. The expected expert route is
  `[1, 58, 68, 240, 20, 24]`.

Implementation:

- Added the stable `rust-star-layers0123-continuous-probe-v1` and
  `rust-star-layers0123-chained-probe-v1` reports plus their CLI commands.
  The synchronized path owns four command buffers and four per-layer KV
  allocations; the chained path commits those buffers without inter-layer
  waits and waits once at the tail.
- Made the command-chain tail explicit in the Rust/Objective-C ABI. The
  existing layers-0/1/2 path declares layer 2, and the new path declares layer
  3. Objective-C validates the declaration, submission order, finalizer, and
  deferred collection against the same tail.
- The first model attempt correctly rejected a nonexistent
  `blk.3.ffn_gate_tid2eid.weight`. DwarfStar and GGUF inspection confirmed
  `deepseek4.hash_layer_count=3`: layers 0 through 2 use token-hash tables,
  while layer 3 uses `blk.3.exp_probs_b.bias` for biased top-k selection.
  The no-copy auxiliary router view and Metal router arguments now select the
  correct mode per layer.
- Updated the standard runtime gate, fixture validation, report writer tests,
  CLI help, and ownership documentation to cover layers 0 through 3.

Target-Mac evidence:

- The synchronized and chained paths both matched every retained FP32 boundary
  for all four layers. Layer 3 selected
  `[1, 58, 68, 240, 20, 24]`, used its own cache allocation, and consumed
  layer 2's live final HC buffer without a host upload.
- Five alternating synchronized/chained pairs remained C0 exact. Synchronized
  wall samples were `[98.490, 97.229, 97.661, 98.371, 100.046]` ms; chained
  samples were `[97.402, 95.409, 95.495, 97.216, 97.566]` ms. Medians were
  98.371 and 97.216 ms: 1.0119x, or 1.17% less wall time for chaining.
- Median summed command GPU intervals were 68.128 ms synchronized and 67.986
  ms chained. The near-equal GPU work and exact outputs support the intended
  interpretation: this measures removed host synchronization inside an
  exhaustive correctness probe, not decoder throughput.
- The complete target-Mac gate passed: 43 Rust tests, optimized macOS build,
  37 Python tests, all 11 differential fixtures, strict validation of 1,288
  required tensors, every standalone Metal probe, synchronized and chained
  four-layer C0 gates, and 30 bit-identical layer-0 steady-state samples. The
  gate's four-layer wall intervals were 99.980 ms synchronized and 96.654 ms
  chained; its layer-0 steady-state median remained 1.500 ms wall.

Decision:

- Accept layer 3, the explicit variable-length chain contract, and the
  hash-to-biased-top-k router transition. Keep the four-layer synchronized path
  as the correctness control and the one-wait path as the scheduler candidate.

Next:

- Move exhaustive boundary readback outside the production timing interval and
  add repeated four-layer execution before extending to another layer.

### 2026-08-15 — Three-layer command chaining remained C0 exact

Objective:

- Remove the two inter-layer host waits from the exact layers-0/1/2 scheduler,
  retain the synchronized path as a control, and measure the result.

Implementation:

- Added `rust-star-layers012-chained-probe-v1` and the
  `layers012-chained-probe` CLI. It commits the same three command buffers to one
  Metal queue, waits once after layer 2, then collects every retained boundary.
- Layer-scoped activation keys preserve layers 0 and 1 until tail comparison;
  per-layer KV allocations remain distinct, and later layers consume the
  preceding live HC Metal buffer through queue ordering. No host upload, event,
  or fixture seam was added.
- Kept `layers012-probe` unchanged as the synchronized C0 control and added the
  chained path to `check_runtime.sh`, the stable JSON writer tests, and runtime
  ownership documentation.

Target-Mac evidence:

- The first chained run matched all three fixtures bit-for-bit with one tail
  wait. All five subsequent alternating control/chained pairs also remained C0
  exact with the expected expert routes.
- Five-pair wall samples were synchronized
  `[82.834, 82.325, 84.160, 81.209, 82.330]` ms and chained
  `[81.481, 79.615, 79.922, 80.175, 81.410]` ms. Medians were 82.330 and
  80.175 ms: 1.0269x, or 2.62% less wall time for chaining.
- Median summed command GPU time was 53.488 ms synchronized and 52.458 ms
  chained. Because this small difference includes ordinary GPU variance and the
  path still performs exhaustive correctness readback, treat the wall result as
  a scheduler diagnostic rather than decoder throughput.
- The final target-Mac gate passed: 40 Rust tests, optimized macOS build, 36
  Python tests, all ten fixture verifiers, strict validation of 1,288 required
  tensors, every standalone Metal probe, synchronized and chained three-layer
  C0 gates, and 30 bit-identical layer-0 steady-state samples. Its synchronized
  wall sum was 83.246 ms and chained wall was 79.899 ms.

Decision:

- Accept the chained lifetime/scheduling boundary and its modest measured wall
  reduction. Retain both paths: synchronized execution is the control, while
  chaining is the exact scheduler candidate. Make no token-throughput claim.

Next:

- Capture and execute the next layer/state boundary through the same ordered
  API, then remeasure chaining over the larger exact slice.

### 2026-08-15 — Layers 0→1→2 matched with explicit per-layer KV ownership

Objective:

- Prove a continuous third-layer handoff, give each executed layer persistent
  cache ownership, and cross the first compressed-attention RoPE boundary.

Oracle fixture:

- Captured all 32 retained layer-2 position-1 boundaries twice from separate
  fresh processes using the pinned DwarfStar executable; all artifacts were
  byte-identical. Captured position-0 `KVcur` independently and derived the
  exact FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer2-complete-v1/`: 28 ordered operations and 33
  tensors totaling 741,808 verified bytes. The expected expert route is
  `[8, 188, 195, 75, 96, 176]`.

Implementation:

- Replaced the executor's reusable cache scratch object with persistent Metal
  buffers keyed by layer identity. The layers-0/1/2 sequence therefore retains
  three distinct KV allocations while carrying one live four-stream HC state
  across the two layer seams.
- Added the stable `rust-star-layers012-continuous-probe-v1` report and
  `layers012-probe` CLI, host tests, fixture validation, documentation, and
  automatic execution in `check_runtime.sh`.
- The first layer-2 run localized its earliest mismatch to `Qcur` after exact
  attention ingress, norm, Q-Lora, Q-Lora norm, and `Qraw` boundaries. DwarfStar
  source inspection showed that layer 2 starts the compressed-attention
  schedule. Parameterizing Q and KV RoPE with base 160,000, scale 1/16,
  original context 65,536, and the matching YaRN factors restored C0 without
  weakening any comparison.

Target-Mac evidence:

- The uninterrupted sequence matched every retained boundary: layer 0 selected
  `[25, 174, 215, 58, 48, 60]`, layer 1 selected
  `[228, 208, 35, 27, 113, 12]`, and layer 2 selected
  `[8, 188, 195, 75, 96, 176]`.
- The canonical run reported 47.226/20.865 ms wall/GPU for layer 0,
  16.310/15.398 ms for layer 1, and 16.401/15.662 ms for layer 2. It used three
  synchronized command buffers and three retained per-layer cache allocations;
  these are diagnostic correctness intervals, not throughput evidence.
- The complete target-Mac gate passed: 39 Rust tests, optimized macOS build, 36
  Python tests, all ten differential fixture verifiers, the cross-language C0
  smoke, strict validation of all 1,288 required tensors, every standalone
  Metal probe, the three-layer gate, and 30 bit-identical layer-0 steady-state
  samples.

Decision:

- Accept explicit per-layer cache ownership and the compressed-RoPE third-layer
  handoff. Keep one synchronized command buffer per layer as the correctness
  control for the next scheduler experiment.

Next:

- Implement and measure a three-layer command-buffer-chained variant, requiring
  the same bitwise checkpoints and ownership report before accepting it.

### 2026-08-15 — Persistent layer-0→layer-1 HC handoff matched DwarfStar

Objective:

- Move setup/execution ownership into the Rust scheduler and prove the first
  real cross-layer state handoff without a host upload or fixture seam.

Oracle fixture:

- Captured all 32 retained layer-1 position-1 boundaries twice from the pinned
  DwarfStar executable; every artifact was byte-identical across fresh
  processes. Captured position-0 `KVcur` separately and derived the exact
  FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer1-complete-v1/`: 28 ordered operations and 33
  tensors totaling 741,808 verified bytes. The fixture binds the pinned source,
  executable, model, prompt, machine, token, layer, and position identities.

Implementation:

- Added a narrow Rust `LayerExecutor` with explicit create/execute/destroy
  lifetime and monotonically ordered layer calls. The Objective-C context now
  rejects model-mapping rebinding and caches exact mmap-backed Metal views plus
  reusable activation buffers.
- Parameterized the proven full-layer chain by layer index. Layer 0 executes
  thirty dispatches; layer 1 executes twenty-eight by skipping embedding and
  repeat and reading layer 0's retained final-HC buffer directly.
- Added a stable `layers01-probe` CLI/JSON contract, fixture and report tests,
  documentation, and automatic target-model gate execution.

Target-Mac evidence:

- Layer 0 retained 25/25 no-copy model-range pointer matches, selected
  `[25, 174, 215, 58, 48, 60]`, and remained C0 exact.
- Layer 1 retained 25/25 pointer matches, selected
  `[228, 208, 35, 27, 113, 12]`, and matched every retained DwarfStar boundary
  bit-for-bit while consuming the live layer-0 HC state.
- The final full gate reported 70.143 ms wall / 30.658 ms GPU for layer 0 and
  26.414 ms wall / 25.457 ms GPU for layer 1. These intervals are diagnostic
  because setup, synchronization, and readback remain included.
- The complete target-Mac gate passed: 37 Rust tests, optimized macOS build, 35
  Python tests, all nine fixture verifiers, the cross-language C0 smoke, strict
  validation of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept the scheduler-owned Metal lifetime and direct HC handoff as the first
  reusable decoder boundary. Keep one synchronized command buffer per layer
  until per-layer KV ownership and a third sequential layer are exact.

Next:

- Add explicit per-layer KV-cache storage and capture/execute layer 2 through
  the same ordered executor API.

### 2026-08-14 — Layer-0 RoPE and first KV-cache write matched DwarfStar

Objective:

- Cross the first stateful decode boundary by finalizing Q/K position encoding
  and proving an exact, isolated cache-row mutation.

Oracle fixture:

- Captured layer 0, decode position 1 in four fresh pinned DwarfStar processes:
  two with standalone diagnostic Q normalization and two with the release
  fused Q-normalization/RoPE path.
- Every repeated artifact was byte-identical. The fused and diagnostic paths
  also produced byte-identical final `Qcur` tensors.
- Added `rust-star/fixtures/layer0-rope-kv-store-v1/` with raw Q/normalized KV
  inputs, diagnostic `Qnorm`, release `Qcur`, pre-store `KVrope`, post-E4M3FN
  `KVcur`, and the documented FP16-rounded cache-row derivation.

Implementation:

- Embedded the pinned DwarfStar `metal/dsv4_rope.metal` and
  `metal/dsv4_kv.metal` sources directly in the runtime Metal library.
- Extended the connected command buffer with fused per-head Q RMSNorm/RoPE,
  layer-0 KV RoPE, and the fused E4M3FN KV finalizer/raw-cache store.
- Added a three-row cache probe initialized with a finite sentinel. The kernel
  targets physical row 1; Rust checks every target-row bit and both neighboring
  guard rows.
- Added a stable CLI/JSON schema, fixture and unit coverage, documentation, and
  automatic execution in `check_runtime.sh`.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 32,768 `Qcur`, 512 `KVrope`, 512 post-FP8 `KVcur`, and 512 cache-row
  values matched the pinned evidence by FP32 bit pattern. Both 512-value guard
  rows remained unchanged.
- The final gate reported 40.305 ms wall and 0.949 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 24 Rust tests, optimized macOS build, 30 Python
  tests, all four fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept the first exact layer-0 cache mutation as the stateful attention
  checkpoint. Add the layer-0 attention scan next, without compression or
  output projection yet.

Next:

- Capture/import the uncompressed layer-0 FlashAttention read over positions
  0..1 and validate the pre-inverse-RoPE attention output.

### 2026-08-14 — Layer-0 Q/K projection setup matched DwarfStar

Objective:

- Continue the real token path beyond Q-Lora through the complete raw Q and
  learned-normalized KV projection setup.

Oracle fixture:

- Captured layer 0, decode position 1 `q_lora_norm`, `KVraw`, `KVnorm`, and
  `Qraw` from the pinned DwarfStar executable in two fresh processes; every
  payload was byte-identical across captures.
- Added `rust-star/fixtures/layer0-qkv-setup-v1/`, retaining the independent
  attention-norm/Q-Lora inputs and all four new boundaries. The fixture contains
  six tensors totaling 159,744 bytes and binds the same source, executable,
  model, prompt, machine, token, layer, and position identities.

Implementation:

- Imported DwarfStar's fused Q-Lora/KV learned RMSNorm, preserving the default
  norm-unification formula and dispatch geometry.
- Extended the connected command buffer with Q8_0 KV projection, fused Q/K
  norm, and Q8_0 Q-B projection. The path now has nine dispatches and ten
  independently page-aligned mmap-backed model views.
- Added a separate stable probe schema/CLI/JSON report, fixture verification,
  unit coverage, documentation, and automatic target-model gate execution.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 1,024 Q-Lora norm values, 512 KV raw values, 512 KV norm values, and
  32,768 raw Q values matched DwarfStar by FP32 bit pattern.
- The final gate reported 38.399 ms wall and 0.473 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 22 Rust tests, optimized macOS build, 29 Python
  tests, all three fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept raw Q plus normalized KV as the next connected layer checkpoint. Keep
  head normalization/RoPE and cache mutation separate so the first stateful
  boundary has its own exact evidence.

Next:

- Capture/import Q head RMSNorm and RoPE plus KV RoPE, then validate the first
  layer-0 cache-row write without adding the attention scan yet.

### 2026-08-14 — Connected layer-0 attention ingress matched DwarfStar

Objective:

- Replace the isolated projection input with a real token embedding and make
  kernel/layer/decode-step evidence independently verifiable.

Artifact contract and oracle fixture:

- Added `rust-star-differential-fixture-v1` with kernel, layer-segment, and
  decode-step scopes, ordered operations, uniquely named tensor boundaries,
  safe relative paths, exact FP32 shapes/encodings, finite-value checks, byte
  counts, and SHA-256 verification.
- Migrated the Q8 projection fixture to the new schema and added a standalone
  verifier that the full runtime gate applies to every committed fixture.
- Captured layer 0, decode position 1 from the pinned DwarfStar executable in
  two fresh processes. All mixer, HC split, collapsed HC, attention norm, and
  Q-Lora payloads were byte-identical across captures.
- Retained seven tensor artifacts totaling 37,056 bytes under
  `rust-star/fixtures/layer0-attention-ingress-v1/`. The fixture records token
  201 and binds the pinned source/tree, executable, model, prompt, machine, and
  graph-dump selection.

Implementation:

- Imported DwarfStar's F16 gather, HC repeat, plain RMSNorm, vectorized F16
  matvec, fused HC split/collapse/weighted norm, and Q8_0 matvec.
- Generated the Objective-C NSString representation of the auditable Metal
  source at build time, keeping the source readable while retaining a
  dependency-free runtime binary.
- Encoded all six operations into one command buffer. Rust retains ownership
  of the read-only whole-model mmap; the shim independently wraps the six
  page-aligned weight ranges with `newBufferWithBytesNoCopy`.
- Added a stable JSON report, CLI command, tests, documentation, and automatic
  execution in the target-model runtime gate.

Target-Mac evidence:

- All six model views preserved exact mmap pointer identity.
- All 24 HC mixer values, 24 split coefficients, 4,096 collapsed HC values,
  4,096 learned-norm values, and 1,024 Q-Lora values matched DwarfStar by FP32
  bit pattern.
- The final gate reported 37.782 ms wall and 0.102 ms Metal GPU time. It
  includes a standalone correctness boundary and is not a decode-throughput
  claim.
- The complete gate passed: 21 Rust tests, optimized macOS build, 28 Python
  tests, both committed fixture verifiers, cross-language C0 artifact smoke,
  strict validation of all 1,288 required GGUF tensors, and every existing and
  new Metal probe.

Decision:

- Accept the connected attention-ingress segment as the first layer-level
  execution checkpoint. Continue immediately after Q-Lora while preserving
  narrow artifacts and avoiding a premature graph framework.

Next:

- Capture and import Q-Lora normalization/Q-B plus the paired KV setup needed
  for the layer-0 attention core.

### 2026-08-14 — First quantized decode projection matched DwarfStar

Objective:

- Validate quantized model bytes, a real runtime activation, DwarfStar's
  decode reduction order, and Rust/Metal ownership in one isolated operation.

Oracle fixture:

- Ran pinned DwarfStar source `b030961` on the M1 Ultra with one prefill token
  and one greedy decode token, filtering its existing graph dump hooks to layer
  0, position 1, `attn_norm` and `q_lora`.
- Retained the 4,096-value input and 1,024-value output under
  `rust-star/fixtures/q8-attn-q-a-v1/`. The manifest binds source tree, model
  SHA-256, machine, prompt, layer/step, tensor, dispatch geometry, and artifact
  SHA-256 values.
- Repeated the complete one-token DwarfStar capture in a fresh process; both
  activation and output files were byte-identical to the retained fixture.

Implementation:

- Imported DwarfStar's `kernel_mul_mv_q8_0_f32` arithmetic and its default four
  simdgroup/two-row dispatch behind the narrow Objective-C ABI.
- Reused the Rust-owned read-only `MAP_SHARED` model mapping and wrapped only
  the page-aligned `blk.0.attn_q_a.weight` span with
  `newBufferWithBytesNoCopy`; the runtime activation uses one small shared
  buffer.
- Added strict fixture/tensor/dispatch validation, bitwise output comparison,
  stable privacy-limited JSON, CLI support, fixture hash tests, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- The Q8_0 tensor begins at byte 79,060,632,640 and contains 4,456,448 bytes.
  Metal wrapped a 4,472,832-byte page-aligned view with a 1,088-byte inner
  offset and exact pointer identity.
- All 1,024 output FP32 bit patterns matched DwarfStar. Input/output checksums
  are `6001855774483604828` and `13770952831385691371`.
- The final standalone dispatch reported 3.990 ms wall and 0.031 ms GPU time;
  pipeline/boundary setup makes this correctness evidence, not decode
  throughput.
- The complete gate passed: 20 Rust tests, optimized macOS build, both no-copy
  model kernels, Metal dispatch/shared-buffer validation, 26 Python tests,
  cross-language C0 artifact smoke, and strict validation of all 1,288 required
  GGUF tensors.

Decision:

- Accept the no-copy Q8_0 matvec as the first runtime-activation arithmetic
  boundary. Build the next increment toward a minimal layer-0 chain and extend
  the artifact format at each new boundary.

Next:

- Specify the kernel/layer artifact envelope around this fixture, then derive
  layer-0 `attn_norm` from the token embedding and feed it into the validated
  projection.

### 2026-08-14 — No-copy model span and first DwarfStar kernel validated

Objective:

- Cross the first real-model execution boundary without adding a graph,
  allocator framework, or whole-model scheduler.

Changes:

- Added a dependency-free Rust `MappedModel` that parses the GGUF and owns one
  read-only `MAP_SHARED` mapping for its complete lifetime.
- Added a page-aligned Objective-C wrapper using
  `newBufferWithBytesNoCopy`; it requires the returned Metal contents pointer
  to equal the mmap pointer and releases the buffer before Rust can unmap.
- Imported DwarfStar's `kernel_get_rows_f16` argument layout, kernel body, and
  threadgroup geometry. Its pipeline is compiled lazily so the original
  dispatch probe's compile-time metric remains comparable.
- Added exact Rust F16-to-F32 reference conversion, bitwise comparison, a stable
  `rust-star-f16-embedding-probe-v1` JSON artifact, CLI support, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- Strict validation selected `token_embd.weight` at byte 77,928,033,088 with
  1,059,061,760 payload bytes. Metal wrapped a 1,059,078,144-byte page-aligned
  view with an 11,072-byte inner offset and exact pointer identity.
- Five rows spanning token IDs 0 through 129,279 produced 20,480 FP32 values;
  every bit matched the CPU reference. The retained checksum is
  `4028716204944876325`.
- The final validation dispatch reported 17.632 ms wall and 0.020 ms GPU time.
  This includes a cold standalone boundary and is correctness evidence, not a
  throughput claim.
- The complete runtime gate passed: 18 Rust tests, optimized macOS build, Metal
  dispatch/shared-buffer validation, 25 Python tests, cross-language C0 smoke,
  strict real-GGUF inspection, and the new no-copy gather.

Decision:

- Accept this mmap/Metal lifetime split as the first model ownership boundary.
  Keep the next increment to one decode projection/matvec before designing the
  graph scheduler.

Next:

- Capture or derive an exact DwarfStar fixture for one Q8_0 decode projection,
  then run the same weight span and activation through Rust Star.

### 2026-08-14 — Target-Mac bootstrap and oracle-v1 completed

Objective:

- Validate the Rust/Metal boundary and pin the DwarfStar oracle on the target
  M1 Ultra before implementing model execution.

Actions and evidence:

- Created a dedicated `agent/rust-star-bootstrap` worktree so the DSpark
  observability branch can continue in a separate Codex task.
- Installed the missing standard `rustfmt` component and ran the full M-002
  gate on the M1 Ultra.
- Fixed the target recipe for `indexer.attn_q_b.weight` from Q8_0 to F16 and
  added a regression test covering every ratio-4 indexer layer.
- Added `pipefail` to the documented evidence command after the first strict
  inspection failure was masked by `tee`.
- Passed 15 Rust tests, 25 Python tests, the optimized macOS build, the C0
  cross-language artifact check, shared-buffer validation, and strict real-GGUF
  inspection of all 1,288 required tensors.
- Captured and independently verified the complete 2K/32K `oracle-v1` bundle.
  Correctness, full-vocabulary frontier logits, and all six timed measurements
  passed. The archive SHA-256 is `5115f445...a09ee3c`.

Decision:

- Accept the quick capture as `oracle-v1`. Keep the extended context capture
  ready but move the critical implementation path to no-copy model mapping and
  the first isolated kernel.

Next:

- Implement and test the no-copy, page-aligned read-only GGUF span boundary,
  then import one decode-critical DwarfStar kernel behind C0 fixtures.

### 2026-08-09 — Initial Rust/Metal ownership and dispatch boundary added

Objective:

- Turn the host-only scaffold into the smallest testable Metal runtime boundary
  without claiming model inference or performance before the M1 Ultra run.

Changes:

- Added a macOS-only Objective-C ARC shim, compiled by Cargo through the Xcode
  toolchain, that owns the Metal device, queue, runtime-compiled probe pipeline,
  shared buffer, command encoders, synchronization, and GPU timestamps.
- Added a safe Rust lifecycle/configuration/reporting layer and `metal-probe`
  command. It compares synchronized per-dispatch command buffers with one
  batched command buffer, validates every resulting element against a CPU
  reference, and can atomically emit a privacy-limited stable JSON artifact.
- Kept non-macOS host contracts buildable through an explicit unsupported stub,
  added a macOS CI compilation job, and made the target-Mac check script run the
  probe before model inspection.
- Applied rustfmt to the complete pre-existing Rust crate when a temporary
  formatter became available; the GGUF/target diffs are formatting-only.
- Documented the ownership boundary in `rust-star/runtime/METAL.md` and extended
  manual task M-002 to request the resulting probe artifact.

Validation in this workspace:

- `check_runtime.sh` passed without a model using a temporary Rust 1.85
  toolchain: rustfmt, 15 Rust unit tests, optimized host build, all 25 Python
  contract tests, and the Rust-writer/Python-reader C0 smoke check succeeded.
- Python compilation, workflow YAML parsing, shell syntax, an independent Rust
  grammar parse, and `git diff --check` passed. The non-macOS `metal-probe`
  command failed explicitly as designed.
- One initial optimized link in the generated in-repository target directory
  reported unresolved thin-LTO symbols; an immediate rebuild and the complete
  check script then passed. Preserve macOS CI and clean-target evidence rather
  than treating this temporary-toolchain anomaly as target evidence.
- This environment still has no Xcode/Metal device or model. The Objective-C
  shim has not been compiled, and no Metal execution, model correctness run, or
  performance measurement has been performed here.

Next:

- Compile and run the complete check script on the M1 Ultra. If the boundary
  validates, add a no-copy read-only model span and one isolated DwarfStar
  kernel while preserving C0 differential hooks.

### 2026-08-09 — Checkpointed paired runner and Rust-side contract added

Objective:

- Complete the engine-neutral A/B orchestration before a Rust Star decoder is
  available, so target-Mac measurements cannot drift into an ad hoc procedure.

Changes:

- Added `rust-star/ENGINE_MEASUREMENT_FORMAT.md` as the shared fresh-process
  contract. DwarfStar implements it now; the Rust Star benchmark must emit the
  same metric semantics directly or through a thin adapter.
- Added `rust-star/paired_runner_lib.py` and `run_paired_benchmark.py` with an
  immutable SHA-bound plan, atomic `state.json` checkpointing after every engine
  process, untimed per-engine warm-ups, alternating A/B and context ordering,
  and optional one-pair-at-a-time execution.
- Timed failures block without automatic retries. Retrying requires an explicit
  reason, reruns the complete pair, retains the old attempt/evidence, and
  preserves already-finalized outputs under `superseded/`.
- Adapter invocations use argument arrays without a shell and fresh process
  groups. Timeouts and interrupts terminate the process group before control
  returns; a hard-interruption checkpoint remains blocked for deliberate review.
- Finalization occurs only after every predeclared pair has one final valid
  attempt. It re-hashes and revalidates every successful engine measurement,
  emits the strict raw artifact, revalidates that artifact, and writes the
  paired summary.
- Added `rust-star/PAIRED_RUNNER.md` and manual task M-005 for the eventual first
  target-Mac paired run. Local plans may contain private paths and are never
  copied into the shared result directory.

Validation:

- All 25 Python contract tests passed. Synthetic end-to-end tests cover pause
  and resume, engine/context ordering, blocked failures, explicit full-pair
  retry, plan immutability, measurement-tamper rejection, validated final
  export, and preservation of superseded results.
- Python compilation, both new CLI help paths, workflow/shell static checks,
  and `git diff --check` passed.
- No real engine, model, Metal, correctness, or performance run was possible in
  this environment.

Next:

- Implement the Rust Star measurement producer alongside the first runnable
  decoder, then instantiate a private paired plan from the completed oracle
  manifest rather than inventing identities in advance.

### 2026-08-09 — Isolated DwarfStar measurement boundary added

Objective:

- Make the pinned oracle measurable through the future paired-runner boundary
  without modifying DwarfStar or waiting for a Rust Star decoder.

Changes:

- Added `rust-star/measure_dwarfstar.py` and
  `dwarfstar_measurement_lib.py`. Each invocation runs one fresh `ds4-bench`
  process at one frontier, preserves its CSV and sanitized logs, and emits a
  checksummed `rust-star-engine-measurement-v1` record.
- Added external complete-process wall time and per-child peak RSS collection.
  Prefill, full-generation, and steady-generation durations are reconstructed
  from DwarfStar's reported counts/rates; the residual is explicitly named
  process overhead rather than inaccurately calling it model-load time.
- Documented a DwarfStar semantic trap: at a terminal single frontier,
  `kvcache_bytes=0` means no serialized session snapshot was created. The
  normalized contract now records `null`, not zero live KV usage.
- Hardened the adapter against missing/partial CSV output, partial generation,
  non-finite metrics, non-empty result directories, timeouts, and private path
  leakage in captured logs.
- Added `rust-star/MEASUREMENT_ADAPTER.md` and included the adapter in the host
  workflow path filters.

Validation:

- Python compilation, CLI help, shell syntax, workflow YAML parsing, and
  `git diff --check` passed.
- All 20 Python contract tests passed. Synthetic child-process tests cover CSV
  normalization, per-process measurements, path redaction, and persisted
  adapter-validation failures.
- No real DwarfStar/model/Metal measurement was run in this environment.

Next:

- Define the Rust Star adapter against the same single-process schema, then add
  a checkpointed paired orchestrator that enforces A/B and context order.

### 2026-08-09 — Manual ledger and paired benchmark v1 defined

Objective:

- Preserve every user/hardware/account-dependent handoff in one actionable
  ledger and remove benchmark-method ambiguity before measurements begin.

Changes:

- Added `RUST_STAR_MANUAL_TASKS.md` with status rules, security boundaries,
  exact commands, dependencies, evidence requirements, and success conditions
  for GitHub Actions approval, target-Mac runtime/model inspection, quick and
  extended oracle capture, and the deferred secure remote-access decision.
- Added `workflow_dispatch` to the host-contract workflow so it can be run from
  GitHub after fork Actions are enabled or approved.
- Added `rust-star/BENCHMARK_PROTOCOL.md` as
  `rust-star-paired-benchmark-v1`. It fixes C0 eligibility, batch-one closed-loop
  decode semantics, 256K primary context, development/full context sets,
  committed-token metrics, alternating paired order, repetition counts,
  thermal controls, raw-data retention, pairwise speedup aggregation, and
  capacity/failure treatment.
- Added `rust-star/PAIRED_RESULT_FORMAT.md`, a strict standard-library parser,
  `summarize_paired_benchmark.py`, and synthetic tests for machine-readable raw
  pairs, host/correctness manifests, exact build/runtime identities,
  predeclared coverage/order, invalid-pair retention, operational metrics, C0
  headline eligibility, and within-pair speedup aggregation.
- Updated the durable project open item to distinguish defining the protocol
  from executing it and collecting the first DwarfStar numbers.

Validation:

- Python compilation passed for the paired parser, CLI, and tests.
- All 16 Python contract tests passed, including cross-context schedule and
  invalid-retry cases.
- `git diff --check` and workflow/shell static checks passed.
- No model, Metal, correctness, or performance run was possible in this
  environment.

Next:

- Define the narrow benchmark-runner interface that will produce paired raw
  rows from DwarfStar and the future Rust executable without coupling either
  engine to the aggregator.

### 2026-08-09 — Strict Rust host-runtime scaffold added

Objective:

- Establish a narrow, executable Rust boundary that can be prepared without
  Metal or model access and prevents unsupported models from entering the
  future engine accidentally.

Changes:

- Added a dependency-free Rust 2021 crate at `rust-star/runtime/` with a
  committed lockfile and a minimum Rust version of 1.74.
- Added a bounded GGUF v3 parser that reads metadata/tensor directories without
  reading tensor payloads. It rejects duplicate names, unknown types, malformed
  booleans/UTF-8, excessive counts/strings, rank/size arithmetic overflow,
  invalid alignment, payload overlap, and out-of-file ranges.
- Derived and encoded the exact DwarfStar Flash shape: 43 layers, width 4096,
  vocabulary 129280, 64 attention heads, 256 experts with 6 active, and the
  model's compression schedule and semantic metadata.
- Added the initial resident imatrix-Q2 recipe validator: IQ2_XXS routed gate/up,
  Q2_K routed down, Q8_0 attention/shared/output, and F16 HC/compressor/indexer.
- Kept checkpoint identity honest: GGUF name metadata is informational; the
  completed oracle's whole-model SHA-256 remains necessary to prove `0731`.
- Added a full-FP32-logit writer with finite-value checks, nine-significant-digit
  output, signed-zero preservation, and lowest-token-ID argmax tie semantics.
- Added `rust-star/check_runtime.sh` to run formatting, Rust unit tests, release
  build, existing Python artifact tests, a Rust-writer/Python-reader C0 smoke
  check, and optional strict model inspection on the target Mac.
- Added a path-filtered GitHub Actions job pinned to Rust 1.74 so formatting,
  compilation, unit tests, release build, and the cross-language artifact smoke
  check can be validated before target-Mac access is available.

Validation in this workspace:

- All Rust files passed an independent Rust grammar parse.
- Shell syntax and `git diff --check` passed.
- All nine existing Python artifact/bundle tests passed.
- The check script correctly exits with a clear Rust-toolchain prerequisite in
  this container.

Not validated here:

- This environment has neither `rustc` nor Cargo, so type-checking, rustfmt, and
  the new Rust unit tests are pending the target-Mac run.
- No real GGUF was available. The first strict inspection may expose a justified
  difference between the actual oracle recipe and the currently derived
  DwarfStar template policy; any change must be evidence-driven and journaled.
- No Metal code, inference, correctness run, or performance measurement exists
  in the scaffold yet.

Next:

- Run `./rust-star/check_runtime.sh /absolute/path/to/model.gguf` on the M1
  Ultra, then run the quick `oracle-v1` capture.

### 2026-08-09 — Offline oracle validation and C0 comparator added

Objective:

- Progress the correctness infrastructure without access to the M1 Ultra or
  model, and give the future Rust runtime an exact output contract.

Changes:

- Added `rust-star/ARTIFACT_FORMAT.md` as the stable initial bundle and
  post-prefill full-logit contract.
- Added `rust-star/artifact_lib.py` with safe bundle extraction, artifact hash
  verification, strict finite-FP32 parsing, bit-pattern comparison, and drift
  metrics.
- Added `rust-star/verify_oracle_bundle.py` to validate a result directory or
  `.tar.gz`, including its sibling/explicit archive SHA-256.
- Added `rust-star/compare_logits.py`; it exits zero only for C0 by default and
  reports mismatch/ULP/error, cosine, KL, argmax, and top-k diagnostics for
  valid non-C0 artifacts.
- Added `rust-star/tests/test_artifact_tools.py` with synthetic fixtures for
  exact equality, signed zero, drift, metadata, tampering, partial manifests,
  and archive traversal.
- Updated the capture README and made tar extraction behavior explicit across
  supported Python versions.

Validation:

- Python compilation succeeded for the capture, library, both CLIs, and tests.
- Nine unit tests passed with deprecation warnings promoted to errors.
- CLI integration checks confirmed C0 exit 0, drift exit 1, and bundle/archive
  verification with the sibling checksum.
- `git diff --check` passed.
- No Metal, model, or performance run was attempted in this environment.

Next:

- Begin a strict, platform-independent Rust runtime scaffold while target-Mac
  capture remains pending.

### 2026-08-09 — Target-Mac oracle capture kit prepared

Objective:

- Prepare everything that can be cloned and executed on the M1 Ultra without
  granting this environment remote access to the machine.

Changes:

- Added `rust-star/capture_oracle_v1.py`, a one-command Python standard-library
  capture tool.
- Added `rust-star/README.md` with quick, extended, custom-context, reporting,
  and security instructions.
- Git-ignored local `rust-star/results/` and `rust-star/.work/` outputs.
- The tool hashes the entire GGUF; captures allowlisted hardware, macOS,
  compiler, Metal, power, and load metadata; exports pinned upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406`; and builds it in a temporary
  directory.
- It runs `--metal-kernels` and `--logprob-vectors`, captures full post-prefill
  FP32 frontier logits separately from timed runs, and aggregates repeated
  batch-one decode/prefill measurements with median, MAD, minimum, and maximum.
- The default quick set is 2K/32K with three repetitions. The opt-in full set is
  2K, 32K, 128K, 256K, 512K, and 1M.
- Result manifests are updated after each completed gate/run so partial evidence
  survives interruption or a long-context capacity failure.
- The shareable archive excludes the model, full environment, unrelated process
  listings, credentials, absolute model path, and Mac serial/UUID identifiers.

Validation in this Linux workspace:

- `python3 -m py_compile rust-star/capture_oracle_v1.py`
- CLI help and default/full dry-run plans.
- Invalid duplicate-context rejection.
- Median/MAD aggregation helper checks against synthetic rows.
- Pinned source commit/tree validation, offline `git archive` extraction,
  deterministic 1M prompt expansion, and command-path sanitization.
- `git diff --check` and ignore-rule verification.

Not validated here:

- No Metal build, GGUF load, correctness gate, logit dump, or performance run
  was possible because this workspace is not the target Mac and has no model.

Next:

- Run the documented default capture on the M1 Ultra and return its archive and
  SHA-256 for inspection before attempting the extended context set.

### 2026-08-09 — Fork synchronized and research branch published

Objective:

- Publish the bootstrap work after correcting the GitHub App installation
  scope.

Actions and evidence:

- Confirmed a new personal-account installation for `deathcoder` was visible
  alongside the existing `dion-labs` installation.
- Fast-forwarded `deathcoder/ds4:main` to upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` without force.
- Created remote branch `agent/rust-star-bootstrap` from that commit.
- Recreated the three-file local documentation tree atomically through GitHub's
  blob, tree, commit, and ref APIs. The initial published commit was
  `6daf28c3faee652463fe0957397e023e4ae7fa98`; its tree
  `e9761869fc0a45fc3880ff42d133deb9c0da2371` exactly matched the local tree.
- No pull request was opened because branch publication, not upstream review,
  was requested.

Next:

- Capture the oracle manifest and baseline on the target M1 Ultra, then define
  the differential golden-artifact format.

### 2026-08-09 — Publication 403 root cause identified

Objective:

- Determine why other GitHub-connected sessions can publish while writes to
  this fork consistently fail.

Evidence:

- The authenticated GitHub user is `deathcoder`.
- The app installation/account inventory contains only organization
  `dion-labs` (installation `152207005`).
- There is no GitHub App installation for the personal `deathcoder` account.
- `deathcoder/ds4` repository metadata reports the user's collaborator
  permission (`push: true`), but Git database writes execute through an app
  installation token. No installation token is scoped to this personal fork.

Conclusion:

- The 403 is not caused by the branch name, upstream commit, or use of the
  low-level ref endpoint. The app is installed on the wrong owning account for
  this repository. Successful sessions likely target repositories covered by
  the `dion-labs` installation or use separately authenticated local git.

Next:

- Install the GitHub app on the `deathcoder` personal account and grant it
  access to `deathcoder/ds4`, then retry ref creation/update through the app.

### 2026-08-09 — GitHub app publication retry

Objective:

- Retry fork synchronization and branch publication through the connected
  GitHub app, and preserve that app as the preferred publication mechanism.

Evidence:

- The app successfully read `deathcoder/ds4` and reported `admin: true` and
  `push: true` repository permissions.
- It successfully resolved upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` through the fork.
- Updating `refs/heads/main` to that commit failed with GitHub HTTP 403:
  `Resource not accessible by integration`.
- Creating `agent/rust-star-bootstrap` from that commit failed with the same
  GitHub HTTP 403 from the create-reference endpoint.

Decision:

- Future sessions should try the GitHub app first, but must confirm an actual
  ref write succeeds. Reported repository permission metadata alone is not
  sufficient evidence that the installation token can publish.

Next:

- Refresh or reconnect the GitHub app installation with repository Contents
  write access, then retry the two ref operations before recreating the local
  documentation commit on the remote branch.

### 2026-08-09 — Bootstrap branch and durable project memory

Objective:

- Synchronize the fork work with upstream, create an isolated research branch,
  and preserve the brainstorming decisions across handoffs and compaction.

Repository actions:

- Cloned `deathcoder/ds4` as `origin`.
- Added `antirez/ds4` as `upstream`.
- Verified the fork's `main` was 137 commits behind and 0 ahead of upstream.
- Fast-forwarded local `main` from `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
  to upstream `b0309611041655f4e45671cfd9c9886aff161406`.
- Created local branch `agent/rust-star-bootstrap` at that upstream commit.
- Added `RUST_STAR_PROJECT.md`, this journal, and an `AGENT.md` pointer that
  requires future Rust Star sessions to read and update them.

Decisions recorded:

- Target only the 48-GPU-core, 128 GB M1 Ultra Mac Studio and
  DeepSeek-V4-Flash-0731 initially.
- Start from the resident imatrix Q2 model and a minimal Rust decoder/benchmark.
- Require C0 full-FP32-logit bit equivalence in the default path.
- Version the upstream DwarfStar oracle rather than treating a moving branch as
  immutable truth.
- Make batch-one committed decode t/s primary; retain prefill, TTFT, memory,
  long-context, low-concurrency, and agent-loop metrics as research scope.
- Deprioritize speculative decoding while retaining the existing DSpark branch
  as evidence and a possible later source of proven mechanisms.
- Use 256K as an important operating point and test correct scaling toward 1M
  rather than creating an undefined separate capacity mode.

Validation:

- No build, model correctness test, or performance benchmark was run. This
  change contains project documentation only.

Blockers/notes:

- The connected GitHub integration could read the repositories but returned
  HTTP 403 for branch/ref writes. Local git work proceeded instead.
- A non-interactive `git push origin main:main` also failed because this
  workspace has no HTTPS GitHub username/token configured. Remote fork
  synchronization and branch publication remain pending write credentials.
- SSH was not a viable fallback: no authentication agent was available and the
  workspace's restricted network could not resolve GitHub on the SSH path.

Next:

- Publish the already committed local state from a GitHub-authenticated
  environment, then capture the oracle manifest and baseline on the target Mac.
