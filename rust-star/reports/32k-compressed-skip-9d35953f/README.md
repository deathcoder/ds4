# Native compressed run-skip: full 32K correctness and component check

Candidate `9d35953f54d4c1c5c8c85ceebf696d1432025c92` passes all three full-output 32K/128 C0 runs,
checking **50,031,360 FP32 words**, and passes the prospectively declared
component selection. Retain it as a correctness-validated candidate.
**This is not an accepted engine speedup or a comparison against DwarfStar.**

Compressed-attention attributed GPU time changes from
29.100284s to 27.131982s
(6.76% lower). The control-normalized aggregate
ratio is 0.950530, and the median of 140 matched
normalized ratios is 0.947557.
The saved 1.968302s is only
0.828% of the baseline recorded GPU
stage sum. That fraction is diagnostic context, not an end-to-end prediction.

## Scope and prior evidence

Starting worktree commit: `2f06a2e8`. Frozen comparator:
`1711856ea309f70877102d15facb4e1243a9fdcc`, executable SHA-256
`a174c1e06ff399056b5946bbac1a34583e43ec7aaac0b078c280e4f04a81f3c8`.
Candidate executable SHA-256: `631b4844599d814ebfcae6706d26c0854169d4573feed86391c41880f71cfd99`.

The Rust-owned `compressed_prefill.metal` is the isolated candidate from
[the run-skip study](../compressed-skip-runs-fbbea533/README.md) with symbol
renaming only. It scans four zero block markers at a time, handles scalar
leftovers, and consolidates mask-pointer advances. The original active-block
arithmetic, grid and partial padded tail remain intact. The prior study's
119,150 integer traces, 120 GPU edge cases and controlled isolated timing are
reused as prior evidence, not claimed as rerun here.

The new source is appended only to the attention-ingress library. Two lazy
NSG8/512 pipeline specializations handle padded/unpadded inputs only when
`prefillLayerChunkStart && position_start == chunk_start && position_start != 0`.
All initial-prefix, legacy continuation and decode dispatches remain unchanged,
as do all imported kernel sources. Removing the scoped additions restores the entire original
host and build files byte for byte; 38 other frozen inputs are unchanged.
The standalone compilation uses the same assembled source, default Metal
compile options and function constants as production. Both pipelines pass.

## Predeclared validation and results

`plan.json` predates the frozen executable and all inference runs. It declares
exactly three fresh processes: normal candidate C0, baseline profiled C0, and
candidate profiled C0. Every run compares all 129,280 logits for prefill plus
128 decode steps against immutable FP32 targets: 16,677,120 words per run.
All output JSON fields, full-logit receipts and decode checksums pass.

| Run | Full C0 | Observed elapsed seconds | Peak process RSS bytes |
|---|---|---:|---:|
| candidate-32k-output | pass | 312.064 | 3,315,138,560 |
| baseline-32k-profiled | pass | 296.182 | 3,333,668,864 |
| candidate-32k-profiled | pass | 276.177 | 3,341,795,328 |

Elapsed times include the correctness harness and startup and are not paired
performance measurements. `/usr/bin/time -l` RSS is a process diagnostic, not
a claim about full model or unified-memory residency. All 320 release Rust
host tests, format check and release build pass. No additional tests, model
runs, retiming or threshold changes were added after results were observed.

Both profiles contain all 287 native attention packets (147 indexed, 140
compressed; 1,883 component intervals) and all 287 tail packets (2,583
intervals). Every interval is strictly ordered, finite and positive; packet
attributions sum to their recorded parents. Initial bootstrap layers 0–2
and all 40 prefix layers are present. `audit.py` recomputes coverage,
component sums, every matched ratio and the selection from the raw profiles.

Selection requires all three ratios below one: raw compressed-attention
aggregate, target/control aggregate, and median matched target/control ratio.
Controls are `attention_output`, `routed_down` and `shared_expert` from the
same 140 compressed-layer tail packets. All three criteria pass.
126/140 normalized packet ratios improve; median absolute
deviation is 0.023656.

| Chunk start | Baseline target s | Candidate target s | Raw ratio | Normalized aggregate | Matched normalized median |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 3.105060 | 3.110732 | 1.001827 | 1.008924 | 0.965582 |
| 8,192 | 3.908060 | 3.443110 | 0.881028 | 0.896134 | 0.898597 |
| 12,288 | 3.686022 | 3.618041 | 0.981557 | 1.002413 | 0.960775 |
| 16,384 | 4.303524 | 3.852324 | 0.895156 | 0.915379 | 0.909903 |
| 20,480 | 4.285185 | 4.032073 | 0.940933 | 0.962358 | 0.960776 |
| 24,576 | 4.798673 | 4.430444 | 0.923264 | 0.948782 | 0.945192 |
| 28,672 | 5.013760 | 4.645260 | 0.926502 | 0.943284 | 0.967175 |

The effect is not uniform across chunks: normalized aggregates are slightly
worse at starts 4,096 (+0.89%) and 12,288 (+0.24%). The preset overall and
matched-median criteria still pass; there was no per-chunk acceptance rule.
The unchanged control sum itself is 1.91%
lower in the candidate process. This broad variation is why the raw 6.76%
component reduction must not be treated as an isolated causal engine gain.

For context, the full stage attribution is below. Components and stages include
encoder splitting, counter collection and fences; they perturb execution.

| Stage | Baseline GPU s | Candidate GPU s | Ratio |
|---|---:|---:|---:|
| initial_bootstrap | 5.532247 | 5.586558 | 1.009817 |
| prefix_layer | 20.451170 | 20.423072 | 0.998626 |
| bootstrap_ingress | 35.855853 | 35.300076 | 0.984500 |
| boundary | 6.681353 | 6.599446 | 0.987741 |
| attention | 65.357604 | 62.626814 | 0.958218 |
| tail | 96.052261 | 93.926144 | 0.977865 |
| ingress | 7.229638 | 7.137823 | 0.987300 |
| compressor | 0.459524 | 0.437580 | 0.952248 |

## Limits and next decision

There is only one baseline/candidate profile pair, always in that order.
Control normalization can expose broad drift but cannot remove every source
of timing variation. The isolated equal-warm A/A method is a separate result;
it is not applied retroactively to these real-model profiles. No unprofiled
engine pair, total-prefill speedup, DwarfStar parity or new frontier is accepted.
Assess these small real-component savings before spending on broader engine
timing; keep any later engine criteria and run count prospective and bounded.

The hardware was checked live as M1 Ultra, 48 GPU cores and 128 GB memory.
Periodic `pmset`, swap and VM observations are preserved in
`host-observations.jsonl`; GPU clocks are not measured. First observed swap:
`vm.swapusage: total = 1024.00M  used = 33.25M  free = 990.75M  (encrypted)`. Last observed swap: `vm.swapusage: total = 1024.00M  used = 33.25M  free = 990.75M  (encrypted)`.
Full model, prompt, oracle source/executables, baseline/candidate executables
and all 41 runtime/imported input hashes pass final verification. The oracle
tracked tree remains clean. Model/GPU work is finished. No push or 256K run.

## Portable evidence

Run `python3 verify.py` here. It verifies the artifact manifest, original/stored
redaction bindings, source restoration and assembly, build/input/executable
bindings, all three full-output records and both strict profiles. The candidate
full host source is losslessly stored as `candidate-host.m.gz`. Binaries and
model weights are omitted; their hashes remain in the manifests. User-specific
paths are replaced by `<repo>`, `<model>` and `<user>` with original/stored
SHA-256 bindings. Replay scripts are provenance, not directly runnable with
those placeholders; the portable verifier needs only Python's standard library.

For reproduction, build the pinned runtime and run
`rust-star long-prefill-chunk-probe "$MODEL" --context 32768 --output-only
--json output.json`; append `--profile-json components.json` for the diagnostic
mode. Use the pinned oracle fixtures and prompt in the source checkout.
