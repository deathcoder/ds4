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
  eligibility, run ordering, aggregation, and capacity semantics; execution
  awaits the target Mac and a runnable candidate. The paired raw/summary JSON
  contract and offline validator/aggregator are implemented and synthetically
  tested. A fresh-process DwarfStar adapter now normalizes one frontier while
  externally recording wall time and peak RSS. The checkpointed paired runner
  enforces warm-up, A/B and context ordering, explicit retries, and final
  validated export through an engine-neutral contract.
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
  captures by bit pattern.
  Full native batched prefill, sparse indexed attention beyond 512 ratio-4
  rows, and the eligible engine-measurement producer remain pending.
- Measurements: Metal batching was 42.861x faster than synchronized submission
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

1. Extend the exact token-ID-to-attention-HC-post native batch tile through the
   layer-0 FFN ingress and batched MoE/HC tail; then broaden the retained row
   and layer coverage toward complete prefill.
2. Add the fixed 512-row ratio-4 indexer top-k and sparse indexed attention so
   128 generated tokens can continue beyond the 2K frontier.
3. Emit the `rust-star-engine-measurement-v1` artifact from the exact
   batched-prefill/128-token loop and connect it to the paired runner.
4. Preserve the four-, six-, eight-, 43-layer, explicit decoder-output, and
   closed-loop diagnostic commands as independently executed controls.
5. Run the extended 2K--1M frontier capture when the Mac can be dedicated to a
   long benchmark; preserve any 512K/1M capacity failure as evidence.
6. Run or approve the fork's GitHub Actions workflow and retain its URL.

## Entries

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
