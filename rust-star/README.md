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
all 129,280 logits despite selecting the same token. The path therefore remains
diagnostic: eligible measurements still require complete native batched prefill
and a full decoder run through the sparse frontier.

A separate retained-state control now seeds the exact layer-2 state immediately
before position 4099 and executes the production retained schedule through its
1,025th compressed row. It matches 16 sparse-boundary tensors bit-for-bit over
54 dispatches with 35/35 no-copy mappings. This closes the first retained sparse
boundary, but not preceding-layer execution, the token-dependent FFN, a
complete decoder, output logits, or throughput.

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
now continues directly through layer 8's complete Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
final HC update. All layer-7 and layer-8 retained boundaries, full attention
outputs, compressor states, and full HC identities match fresh DwarfStar
processes exactly. The complete layers-0--8 command uses 383 dispatches and
preserves 196/196 no-copy model mappings. Exactly 512 ratio-4 compressed rows
remain dense at the prompt boundary. The pinned DwarfStar default remains dense
through 1,024 rows and first switches at 1,025. The position-2051 override
remains an independent one-block control; two fresh production-default captures
at position 4099 now add the exact two-block argsort merge and validate all
1,025 scores, top-512 indices, indexed attention, and inverse RoPE bit-for-bit.
The same first-boundary schedule is wired into retained even-layer state with
35 no-copy model mappings. Execution of a complete decoder through that branch,
output-logit C0, generalization past 1,025 rows, and throughput remain pending.

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
