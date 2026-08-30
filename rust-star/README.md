# Rust Star Oracle Capture

This directory contains the reproducible capture kit for `oracle-v1`. Run it on
the target M1 Ultra before implementing or benchmarking the Rust runtime.

The capture does not need network access, GitHub credentials, SSH keys, or any
other secret. It records only an allowlist of hardware and toolchain fields; it
does not dump the process environment or Mac serial identifiers.

The platform-independent Rust host scaffold now lives in `rust-star/runtime/`.
Before the oracle capture, compile its tests and inspect the model directory:

```sh
./rust-star/check_runtime.sh \
  /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

See `rust-star/runtime/README.md` for its exact scope. It now has the minimal
Metal ownership/dispatch probe and a continuous layers-0/1/2/3 path under one
persistent executor, with live GPU HC handoffs and one retained KV-cache
allocation per layer. All retained boundaries, including layer 2's first
compressed-attention RoPE path and layer 3's transition from hash routing to
biased top-k, match pinned DwarfStar fixtures bit-for-bit. A chained scheduler
variant now submits all four command buffers without
inter-layer host waits and performs one exact tail readback. A separate
persistent four-layer replay prepares all model bindings and fixture buffers
once, excludes warmups, records command-buffer timing without in-interval host
readback, and performs one exhaustive C0 collection after the final sample. The
persistent executor also has a three-step correctness path that advances from
token 201/position 1 through token 361/position 2 to token 1915/position 3,
retains and grows all four raw KV caches from two to four rows, and hands off
layer 3's final 16,384-element HC state. It advances layer 2's ratio-4 attention
and indexer compressors, validates the first FP8 compressed KV row, and also
advances layer 3's non-emitting ratio-128 compressor state. Every retained
boundary in all three steps matches two independent fresh-process DwarfStar
captures bit-for-bit. A six-layer extension uses the same prepared scheduler and
generalized alternating compressor ownership through layers 4 and 5. Layer 4
matches a second ratio-4 compressed KV emission at position 3, while layer 5
advances ratio-128 state without emitting; all new boundaries are independently
captured and C0 exact. A bounded position-127 replay now feeds 128 independently
captured `attn_norm` rows through the layer-3 and layer-5 ratio-128 compressors
and matches both first emitted KV rows bit-for-bit. Those rows are explicit
oracle inputs: the replay performs neither sampling nor a complete decoder pass.
An eight-layer extension continues through layers 6 and 7, validates a third
ratio-4 emission from layer 6, and retains layer 7's ratio-128 state. All 197
new oracle payload pairs were independently repeated and byte-identical. The
persistent executor now also has a checked all-transformer command covering
layers 0 through 42. Its remaining 3,448 oracle payload pairs were captured
twice and matched byte-for-byte before 140 new fixture envelopes were imported.
The command retains 43 raw KV caches, validates every even-layer position-3
ratio-4 emission through layer 42, and hands off layer 42's exact
16,384-element HC state. The four-, six-, and eight-layer commands remain
separate regression controls. The persistent layer-0 gate reuses its pipelines,
25 no-copy model views, cache storage, and activation buffers for repeated
steady-state timing. The deepest exact slice now continues through all 43
layers, full-vocabulary logits, and a position-127 greedy feedback loop. A new
cold-start mode begins with empty Rust-owned raw and compressor state, evaluates
the raw one-token prompt at position 0, matches its full logits bit-for-bit, and
then commits the complete 128-token oracle transcript. It crosses the first
live ratio-128 emissions in layers 3 and 5 and matches both those rows and the
final 129,280 logits. A separate 2K command now grows context-sized compressed
memory, advances a true 128-row raw-KV ring, and exactly matches two fresh
DwarfStar one-token decode replays over the canonical prompt. It also proves
that this sequential construction differs from DwarfStar's batched prefill in
all 129,280 logits despite selecting the same token. That path therefore remains
diagnostic. The exact native path now retains all 43 layers of batched-prefill
state, adopts its raw rings, compressed histories, and recurrent states with
one GPU blit command, and greedily decodes positions 2048--4099. All 2,052
selected tokens and both pinned full-logit frontiers are exact, including the
first production-default 1,025-row sparse ratio-4 step. The timing-specific
engine producer now runs native prefill without diagnostic output collection,
preserves the exact transcript, and passes the fresh-process adapter's
paired-eligibility checks at the initial 2K/128 frontier. Greedy production
sampling appends a lowest-ID GPU top-1 reduction to the output-head command and
reads back only its eight-byte result. The diagnostic paths still transfer and
compare all 129,280 logits, so the C0 boundary is unchanged.

A separate retained-state control now seeds the exact layer-2 state immediately
before position 4099 and executes the production retained schedule through its
1,025th compressed row. It matches 16 sparse-boundary tensors bit-for-bit over
54 dispatches with 35/35 no-copy mappings. This closes the first retained sparse
boundary, but not preceding-layer execution, the token-dependent FFN, a
complete decoder, output logits, or throughput.

The retained sparse scheduler now uses a context-capacity-sized ping-pong
workspace and derives its active sort/merge schedule from visible compressed
rows. A second exact control at position 8195 seeds the prior 127-row raw-cache
histories for layers 0-2 plus layer 2's compressed/recurrent state, but no
incoming HC. It executes layers 0, 1, and 2 normally for token 381, commits
compressed row 2,049, and runs three initial sort blocks plus two merge passes.
All 44 checked boundaries match, including both predecessor cache writes and HC
handoffs, over 113 dispatches with 85/85 no-copy mappings. This proves live
preceding-layer execution and the complete retained sparse layer, while still
leaving the prior cache histories seeded. A complete decoder, output logits,
and throughput remain unclaimed.

The retained boundary now extends through the complete transformer and output
head at position 8195. Two fresh DwarfStar processes produced identical
136,200,592-byte layer-state payloads, all 43 FFN HC handoffs, and all 129,280
logits for token 381. The compact fixture preserves the 127 prior raw rows per
layer, every recurrent compressor state, the full ratio-4 indexer histories,
and only the 512 attention rows selected by each sparse layer. Rust executes
all 43 layers with live HC handoffs, matches 44 pinned outputs bit-for-bit, and
selects token 35,597. This proves a complete retained decoder step with seeded
history and exact output logits; native prefill and throughput remain unclaimed.

The first native
batch boundary is now implemented separately: repeated captures localize the
earliest difference to layer 0's Q8 Q-A projection, and the Rust Metal probe
matches both the 128-row M1 batch kernel and its one-row decode control
bit-for-bit. A second native boundary now runs the final 32-row tile continuously
from token IDs through layer 0's HC ingress, complete Q/KV projection setup,
both KV finalization paths, guarded raw-cache storage, and zero-prefix batched
FlashAttention, inverse RoPE, grouped Q8 attention output, the four-stream
attention HC post-update, FFN ingress, the decomposed M1 batch router, routed
IQ2_XXS/Q2_K experts, the shared Q8_0 expert, and the additive FFN HC tail. It
reconstructs the 2,048-row contiguous KV input from a captured 2,016-row prefix
and the live final tile, matches all 6,979,776 retained produced FP32 values
plus 192 selected expert IDs from repeated DwarfStar captures, and proves cache
rows 0--95 remain untouched. This completes layer 0 for the isolated final
tile, not the full 2K prefill path. Two separate controls continue the same live
command buffer into layer 1: the shorter 47-dispatch Q-A boundary remains
independently executable, while the complete 84-dispatch boundary finishes
layer 1 through attention, routed/shared experts, and the additive FFN HC tail.
The complete command uses 49 no-copy model views and reproduces 12,878,208
produced FP32 values plus 384 selected expert IDs across both layers from
repeated DwarfStar captures. The same executor now also reproduces the previous
tile at positions 1984--2015, including its different RoPE positions,
1,984-row KV prefix, and raw-ring target row 64. Together the two controls cover
64 exact prompt rows. An additional persistent-context control retains both
layers' first-tile KV buffers, validates the retained 2,016-row prefixes against
the oracle, and has the second tile append to and consume those live buffers
without reassembling its execution state from a captured prefix. It remains a
two-command-buffer checkpoint with one inter-tile host wait. A separate 64-tile
loop now starts with empty layer-0/layer-1 KV buffers and advances all 2,048
canonical prompt rows in one persistent context. Every accumulated prefix is
validated bit-for-bit before the next tile, every tile preserves 49/49 no-copy
model views, and the final tile retains its exhaustive output comparison.
The native layer-2 boundary now consumes every one of those live layer-1 output
tiles. Six additional dispatches per tile execute layer 2's HC ingress,
attention normalization, Q-A/KV projections, and fused Q/KV learned norm. A
new repeated full-2K `KVnorm` capture is byte-identical across fresh DwarfStar
processes, and all 1,048,576 runtime values match it bit-for-bit. Every tile
preserves 57/57 no-copy model views. This validates all live layer-1 output rows
through a downstream native boundary. The next exact command continues every
tile through layer 2's YaRN-scaled compressed-attention RoPE and E4M3FN KV
finalization. It retains a live 2,048-row layer-2 KV allocation, validates the
complete prefix before every append, and matches all full-2K KVrope and KVcur
values from two fresh DwarfStar captures. The persistent loop now also owns
both ratio-4 compressors, matches all 512 attention/indexer compressed rows and
their final recurrent states, and retains every normalized layer-2 query row.
A final 32-dispatch command executes Q-B, compressed YaRN, dense mixed
FlashAttention over 2,048 raw plus 512 compressed rows, inverse RoPE, and both
Q8 attention-output projections, then both four-stream HC updates and the full
token-hash routed/shared FFN. Its complete 2048 x 4096 attention output, full
attention-HC state, and full final 2048 x 16384 HC state match two fresh
DwarfStar captures, with exact final-tile intermediate gates and 16/16 no-copy
model views. The custom decomposed router kernels now support the complete
2,048-row batch while preserving the established 32-row schedules.
The same terminal command now continues directly from layer 2's full final HC
buffer through layer 3's four-dispatch HC attention ingress, learned norm, and
Q-Lora projection, then through KV projection, fused Q/KV learned norm, Q-B,
compressed Q/KV RoPE, and FP8 KV finalization. It uses nine additional no-copy
model views, retains all 2,048 layer-3 raw-KV rows in the Metal context, and
matches the repeated DwarfStar final-tile boundaries plus pinned full-2K
checksums exactly without a host HC upload. Four more no-copy views now carry
the layer-3 ratio-128 compressor weights. Its two full-batch F16 projections,
score/APE update, exact 128-row pooling order, learned norm, compressed RoPE,
and FP8 finalization reproduce all 16 prompt emissions and both final recurrent
state tensors bit-for-bit. The persistent context retains the compressed rows
and state for the next boundary. Three final no-copy views then drive layer 3's
dense mixed attention over 2,048 raw plus 16 compressed rows, inverse RoPE, and
both Q8 attention-output projections before the additive four-stream HC post.
The 2,064 logical keys use a fully masked 2,112-row physical extent required by
the 64-row FlashAttention block contract. The complete 2048 x 4096 attention
output and final 32-row HC tile match fresh-process DwarfStar captures exactly;
the full HC identity is pinned by checksum. The same command now continues
through layer 3's FFN HC ingress, learned norm, biased top-6 expert selection,
unbiased router-weight normalization, routed IQ2_XXS/Q2_K experts, the shared
Q8_0 expert, and the additive FFN HC update. Two fresh DwarfStar captures agree
bit-for-bit on every retained final-tile boundary and the complete final HC
identity. The terminal schedule uses 79 dispatches and preserves all 44/44
no-copy model mappings, establishing exact complete layer-3 prefill ownership.
The retained layer-3 final HC now flows directly into layer 4's HC attention
ingress, learned norm, Q-A/KV projections, fused Q/KV learned norm, Q-B,
compressed RoPE, and FP8 KV finalization. All ten final-tile boundaries match
four fresh DwarfStar captures exactly; the full-2K tensor identities are pinned
by SHA-256. The expanded terminal schedule now continues through both layer-4
ratio-4 compressors. Four F16 projections feed the exact replay pool, learned
norm, compressed RoPE, E4M3FN/indexer QAT, and recurrent-state refresh. All 512
attention rows, all 512 indexer rows, and all four final state tensors match
fresh-process DwarfStar captures bit-for-bit while retaining the full layer-4
Q/KV and paired-compressor state for the next boundary. Three more no-copy model
views now drive layer 4's dense mixed attention across 2,048 raw plus 512
compressed rows, inverse RoPE, and both Q8 attention-output projections before
the additive four-stream HC post. The complete 2048 x 4096 attention output
and final 32-row HC tile match two fresh DwarfStar captures bit-for-bit, with
their full identities checksum-pinned. Twelve final no-copy views then carry
layer 4 through FFN HC ingress, learned norm, biased top-6 selection, unbiased
router-weight normalization, routed IQ2_XXS/Q2_K experts, the shared Q8_0
expert, and the additive FFN HC update. Every retained final-tile boundary and
the complete final HC identity match two fresh DwarfStar processes exactly.
The terminal schedule now continues directly from layer 4's final HC into
layer 5's HC attention ingress, learned norm, Q-A/KV projections, fused Q/KV
normalization, Q-B, compressed RoPE, and FP8 KV finalization. All ten layer-5
final-tile boundaries match four fresh DwarfStar captures bit-for-bit. The
full layer-5 Q/KV buffers then feed both F16 compressor projections and the
exact ratio-128 batch path. All 16 compressed rows and both final 128x512
recurrent-state tensors match two fresh DwarfStar processes bit-for-bit. The
same retained state now drives layer 5's dense mixed attention over 2,048 raw
plus 16 compressed rows, inverse RoPE, both Q8 attention-output projections,
and the additive HC post. The complete 2048 x 4096 attention output and final
HC identity match two fresh DwarfStar captures bit-for-bit. The combined
terminal schedule then carries that HC state through layer 5's learned FFN
ingress, biased top-6 routing, routed IQ2_XXS/Q2_K experts, shared Q8_0 expert,
and additive final HC update. Every retained FFN boundary and the complete final
HC identity match two fresh DwarfStar processes exactly. That final state now
feeds layer 6's HC attention ingress, Q-A/KV projections, fused learned Q/KV
normalization, Q-B, compressed RoPE, and FP8 KV finalization. All ten layer-6
final-tile boundaries match four fresh DwarfStar captures bit-for-bit, with
their complete 2K identities SHA-256-pinned. The retained full layer-6 Q/KV
state now feeds both paired F16 ratio-4 compressor projections, replay pooling,
learned normalization, compressed RoPE, attention E4M3FN/indexer QAT, and
recurrent-state refresh. All 512 attention/indexer compressed rows and four
final recurrent-state tensors match two fresh DwarfStar processes bit-for-bit.
The command uses 236 dispatches and preserves 121/121 no-copy model mappings,
establishing complete native layers 0 through 5 plus exact layer-6 paired
compressors at the full 2K boundary. The same retained state now drives dense
mixed attention over 2,048 raw plus 512 compressed rows, inverse RoPE, both Q8
attention-output projections, and the additive HC post. The complete
2048x4096 attention output, three diagnostic rows, final HC tile, and full HC
identity match two fresh DwarfStar processes exactly. The extended command uses
245 dispatches and preserves 124/124 no-copy model mappings. That attention HC
state now feeds layer 6's learned FFN ingress, biased top-6 routing, routed
IQ2_XXS/Q2_K experts, shared Q8_0 expert, and additive final HC update. Every
retained FFN boundary and the complete final HC identity match two fresh
DwarfStar processes exactly. The retained layer-6 final HC then feeds layer 7
through its complete Q/KV state, ratio-128 compressor, dense mixed attention,
biased top-6 routed/shared FFN, and additive final HC update. Layer 7's final HC
continues directly through layer 8's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update. That retained state now continues through layer 9's complete
Q/KV state, ratio-128 compressor, dense mixed attention, routed/shared FFN, and
final HC update, then through layer 10's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update, and then through layer 11's complete Q/KV state, ratio-128
compressor, dense mixed attention, routed/shared FFN, and final HC update.
Layer 11's retained final HC now continues through layer 12's complete Q/KV
state, paired ratio-4 attention/indexer compressors, dense mixed attention,
routed/shared FFN, and final HC update, then through layer 13's complete Q/KV
state, ratio-128 compressor, dense mixed attention, routed/shared FFN, and
final HC update, then through layer 14's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update, then through layer 15's complete Q/KV state, ratio-128
compressor, dense mixed attention, routed/shared FFN, and final HC update, and
then through layer 16's complete Q/KV state, paired ratio-4 attention/indexer
compressors, dense mixed attention, routed/shared FFN, and final HC update, and
then through layer 17's complete Q/KV state, ratio-128 compressor, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 18's
complete Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 19's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 20's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 21's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 22's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 23's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 24's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update. All layer-7 through layer-24
retained boundaries, full attention outputs, compressor states, and full HC
identities match fresh DwarfStar processes exactly. The retained state then
continues through layer 25's complete Q/KV state, ratio-128 compressor, dense
mixed attention, routed/shared FFN, and final HC update. The retained state
then continues through layer 26's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update. The retained state then continues through layer 27's complete
Q/KV state, ratio-128 compressor, dense mixed attention, routed/shared FFN,
and final HC update. The retained state then continues through layer 28's
complete Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update. The retained state then
continues through layer 29's complete Q/KV state, ratio-128 compressor, dense
mixed attention, routed/shared FFN, and final HC update. The retained state then
continues through layer 30's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update. The retained state then continues through layer 31's complete
Q/KV state, ratio-128 compressor, dense mixed attention, routed/shared FFN,
and final HC update, through layer 32's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update, and then through layer 33's complete Q/KV state, ratio-128
compressor, dense mixed attention, routed/shared FFN, and final HC update, and
then through layer 34's complete Q/KV state, paired ratio-4 attention/indexer
compressors, dense mixed attention, routed/shared FFN, and final HC update, and
then through layer 35's complete Q/KV state, ratio-128 compressor, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 36's
complete Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 37's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 38's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 39's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 40's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update, and then through layer 41's
complete Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and final HC update, and then through layer 42's complete
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and final HC update. The complete transformer
layers-0--42 command uses 2,372 dispatches and preserves 1,216/1,216
no-copy model mappings. The same persistent context now applies the exact
five-dispatch output head to the retained final layer-42 row, preserving 5/5
additional no-copy model mappings and matching all 129,280 repeated DwarfStar
batched-prefill logits bit-for-bit. Lowest-ID argmax selects token 15342, so
the complete native model schedule uses 2,377 dispatches and 1,221 mappings.
The retained transformer is no longer limited to that 2K correctness frontier.
The `long-prefill-transformer-probe` command consumes the first 4,096 tokens of
the accepted oracle-v3 32K stream, executes layers 0--42 and the output head,
and retains 4,096 raw rows per layer, 1,024 ratio-4 rows, and 32 ratio-128 rows.
The first target-Mac run preserved all 1,216/1,216 no-copy transformer mappings
across 2,372 dispatches and selected token 565. It measured 5,184.460 ms of
bootstrap GPU time and 14,080.315 ms of transformer GPU time. This is a native
4K schedule and lifetime checkpoint, not a C0 or throughput result; the
separate full 2K regression still matched every retained boundary and all
129,280 logits bit-for-bit and selected token 15342.
An intermediate 8K C0 target is now pinned as
`dwarfstar-oracle-v3-prefill-frontier-8192`. It was captured from the accepted
host-synchronized `d35fb12` producer in four fresh processes; all four complete
JSON logit tensors have SHA-256 `791ee1ea...8f94`, the packed FP32 tensor has
SHA-256 `626454dd...3a1a`, and lowest-ID argmax selects token 77179. The
176.44--181.06 tok/s observations are conformance-only and are not paired
benchmark evidence. `import_frontier8192_fixture.py` revalidates the producer,
executable, model, prompt, token prefix, CSV rows, and all four tensors before
reconstructing the fixture.
The `long-prefill-continuation-bootstrap-probe` now keeps the completed first
4K transformer in one Metal context and advances positions 4,096--8,191
through complete layers 0 and 1 plus layer 2 raw KV and paired ratio-4
compressors. The first target-Mac run completed 64 continuation tiles and
7,556 dispatches, preserved all 4,160/4,160 no-copy mappings, and grew the
retained state to 8,192 raw and 2,048 compressed rows without discarding the
first-chunk prefix or recurrent compressor state. Continuation GPU time was
5,124.933 ms. This is the retained second-chunk bootstrap gate only: layers
2--42, production sparse attention, and exact 8K output logits remain the next
milestone.
Exactly 512 ratio-4 compressed rows for each even layer through layer 42 and
16 ratio-128 rows for each odd layer through layer 41 remain dense at the
prompt boundary. The
pinned DwarfStar default remains dense through 1,024 rows and first switches at
1,025. The position-2051 override
remains an independent one-block control; two fresh production-default captures
at position 4099 now add the exact two-block argsort merge and validate all
1,025 scores, top-512 indices, indexed attention, and inverse RoPE bit-for-bit.
The same first-boundary schedule is wired into retained even-layer state with
35 no-copy model mappings, and a complete retained position-8195 decoder step
now executes all 43 layers through that branch and matches full-vocabulary
logits exactly. Exact sparse post-prompt integration is now complete through
position 4099 via the GPU-only native-prefill handoff. The latest checkpointed
five-pair C0 development comparison at 2K/128 completed without retries or
invalid attempts, and all five pairs favored Rust Star on both generation
measures. Immutable commit `eae11df` measured 23.655165050 tok/s steady versus
DwarfStar's 22.81, a 1.037052391x median pairwise ratio and validated 3.71%
lead. Complete generation measured 23.611303513 versus 22.18 tok/s, a
1.064531267x ratio and 6.45% lead. First-token latency is effectively tied at a
1.013342227x paired ratio. Queuing the exact 64-by-32-row prefill schedule with
one tail wait improved the prefill ratio from `2c6f80e`'s 0.659348296x to
0.673232187x; the remaining prefill gap is 32.68%. This exact development
result is not the protocol's 256K headline claim. The subsequent immutable
`cdb6377` checkpoint widens only the timing-only bootstrap from 64 32-row
command buffers to 32 64-row command buffers. It fixes the row-derived matrix,
FlashAttention, and grouped-output launch geometry that invalidated the first
wide-row experiment while retaining the exact 32-row diagnostic schedule.
`cdb6377` then completed five exact pairs without a failure, retry, or invalid
attempt. Rust Star measured 133.070209951 prefill tok/s versus DwarfStar's
186.35, for a 0.713851042x median pairwise ratio. The remaining prefill gap is
28.61%, down 4.06 percentage points from `eae11df`. Rust retains a 3.87% steady
decode lead and 6.47% complete-generation lead; first-token wall time is 13.0%
slower in this run. This exact development result is not the protocol's 256K
headline claim.

An explicitly ineligible synchronized per-layer diagnostic now attributes the
remaining native prefill transformer interval. Across layers 2--42, Rust's even
layers average 326.577 ms GPU and odd layers 251.904 ms; DwarfStar's independent
split profile averages 283.103 and 223.180 ms respectively. The similar
alternating ratios (1.296x Rust, 1.268x Dwarf) rule out a single pathological
layer. Representative layers 4 and 5 are now split into attention/FFN and five
attention-internal intervals. Three exact Rust profiles and three independent
Dwarf profiles per layer isolate the main 512-wide FlashAttention dispatch as
the only consistent deficit: median Rust time is 140.823 versus 108.419 ms in
layer 4 and 71.147 versus 54.666 ms in layer 5, approximately 30% slower in
both families. Rust's QKV/RoPE, compressor/staging, and output/HC medians are
all at or ahead of Dwarf; Flash block preparation is only about 0.03 ms.

The two runtimes use the same Metal kernel source, grid, threadgroup memory,
fast-math default, and shared scratch storage on this M1 Ultra. DwarfStar's
private-scratch path is M5-only, so storage-mode changes are not supported by
this evidence. Reducing the 512-wide non-vector pipeline and all 43 matching
launches from eight to four SIMD groups preserved the exact transcript and cut
the median representative Flash intervals by 63.4% and 65.4%. Three eligible
runs measured 170.329--172.234 prefill tok/s, with a 171.682 median versus
138.083 for the immediately preceding NSG=8 control, a 24.3% gain. Immutable
commit `0cbffdf` then completed five C0 pairs without a retry or invalid
attempt. Rust measured 163.873 prefill tok/s versus DwarfStar's 182.370, for a
0.897865x median pairwise ratio and a remaining 10.21% gap. The previous
validated ratio was 0.713851x, so NSG=4 closes 18.40 percentage points of the
gap. Steady decode is effectively tied/slightly ahead at 1.001367x and complete
generation is 1.014559x ahead. The profiler's 52 command buffers and waits
remain diagnostic only; the next target is the pre-transformer bootstrap and
residency interval, not transformer attention.

The runtime also exposes `engine-retained-measure` to test that residency as an
honest lifecycle boundary. It runs a fully charged first 2K/128 cycle and a
second exact cycle in the same process and Metal context, requires the second
cycle to reset request-scoped activations and caches while reusing the existing
residency set, and reports the two prompt costs separately. This diagnostic is
never paired-eligible; normal `engine-measure`
continues to require fresh residency so first-load cost cannot disappear from
the benchmark contract.

Project controls and benchmark contracts:

- `RUST_STAR_MANUAL_TASKS.md` is the canonical ledger for work that needs the
  target Mac, model, GitHub UI, or a deliberate access decision.
- `rust-star/BENCHMARK_PROTOCOL.md` defines the paired DwarfStar/Rust Star
  comparison.
- `rust-star/PAIRED_RESULT_FORMAT.md` defines the validated raw and summary JSON
  boundary for future benchmark runners.
- `rust-star/MEASUREMENT_ADAPTER.md` defines the isolated DwarfStar measurement
  boundary used to populate those paired records.
- `rust-star/ENGINE_MEASUREMENT_FORMAT.md` defines the equivalent Rust Star
  boundary, and `rust-star/PAIRED_RUNNER.md` documents resumable A/B execution.
- `rust-star/runtime/METAL.md` defines the initial Rust/Objective-C ownership
  split and correctness-checked command-dispatch probe.

## Quick capture

Clone the published research branch on the Mac Studio, then run:

```sh
git clone --branch agent/rust-star-bootstrap \
  https://github.com/deathcoder/ds4.git rust-star-ds4
cd rust-star-ds4

python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

The default capture uses 2K and 32K context frontiers with three timed
repetitions each. It:

1. hashes the complete model;
2. records a privacy-filtered hardware, macOS, and toolchain manifest;
3. exports and builds an isolated copy of pinned upstream commit `b030961`;
4. runs the existing Metal-kernel and official logprob-vector correctness gates;
5. captures full FP32 frontier logits in separate, untimed conformance runs;
6. runs repeated batch-one DwarfStar benchmarks; and
7. writes raw CSVs, median/MAD summaries, logs, hashes, and a shareable archive.

Model hashing reads the entire GGUF and will take a while. The model is never
copied into the results or archive. The build uses a temporary source directory
and a generic `model.gguf` symlink so local account paths do not appear in the
logit artifacts.

At completion the script prints paths similar to:

```text
Results: rust-star/results/oracle-v1-YYYYMMDDTHHMMSSZ
Archive: rust-star/results/oracle-v1-YYYYMMDDTHHMMSSZ.tar.gz
SHA-256: ...
```

Send back the `.tar.gz` archive and its printed SHA-256. Capture outputs are
git-ignored; do not force-add them to the repository.

You can verify the completed archive before sending it:

```sh
python3 rust-star/verify_oracle_bundle.py \
  rust-star/results/oracle-v1-TIMESTAMP.tar.gz
```

The verifier automatically uses the sibling `.sha256` file. See
`rust-star/ARTIFACT_FORMAT.md` for bundle and exact-logit comparison semantics.

## Extended contexts

After the quick capture succeeds, run the extended frontier set with five
repetitions:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/model.gguf \
  --full \
  --repetitions 5
```

`--full` covers 2K, 32K, 128K, 256K, 512K, and 1M. The 512K and 1M runs may
approach the safe working-set limit on a 128 GB machine. Run the quick capture
first and close other memory-heavy applications before the extended capture.

For a custom subset:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/model.gguf \
  --contexts 2048,32768,131072,262144 \
  --repetitions 5 \
  --notes "Room 22C; normal background services; no other model processes"
```

Useful options:

- `--output DIR`: choose an empty output directory.
- `--gen-tokens N`: change the timed greedy decode length; default 128.
- `--skip-correctness`: omit the two existing correctness gates.
- `--skip-conformance`: omit full-logit artifacts.
- `--skip-performance`: collect only manifest/correctness/conformance data.
- `--dry-run`: print the plan without hashing, building, or running the model.

Skipping a section is recorded in the manifest and does not produce a complete
`oracle-v1` capture.

## Artifact semantics

Conformance and performance runs are deliberately separate. The headline
throughput CSVs are produced without logit dumping. Full frontier-logit JSON
uses DwarfStar's nine-significant-digit FP32 encoding, which round-trips finite
FP32 values exactly.

The first capture covers full logits immediately after prefill at each selected
context. A later differential harness will extend this to kernel/layer
boundaries and every decode step. The manifest states this scope explicitly so
the initial artifact is not mistaken for complete engine conformance coverage.

Performance runs use greedy decode while excluding EOS, matching `ds4-bench`.
Each selected context runs in its own process and therefore measures prefill
from zero to that frontier. Repetition order alternates ascending and descending
contexts to reduce systematic thermal-order bias.

## Security boundary

The archive contains model filename, byte size, and SHA-256, but never model
contents. It excludes:

- environment variables not on the small compiler/runtime allowlist;
- GitHub tokens, SSH configuration, and credentials;
- process listings and command lines from unrelated applications;
- hardware serial number, UUID, and provisioning identifiers; and
- the absolute model path.

Review free-form `--notes` before sharing because their content is supplied by
you. A future remote-execution setup should use a private network and a
short-lived, command-restricted credential; it is intentionally outside this
capture kit.
