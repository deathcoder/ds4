# 32K routed-prefill grid-order validation

Candidate `4552043579bb86b6f94e682e3a154937322169ee` versus frozen Rust baseline `d6c14fa0918a5130da72aa7be6e56b82696fb8b0`, 2026-09-11, Apple M1 Ultra (48 GPU cores, 128 GB).

## Decision

All three complete 32K/128 C0 runs pass. **Retain as a correctness-validated candidate for later engine timing** under the predeclared selection rule. This is a single diagnostic comparison with deliberately perturbed profiling, not accepted engine performance. No Rust-versus-DwarfStar throughput result is updated, no longer frontier is validated, and 256K remains paused.

Routed gate/up totals change from 30.856877s to 30.425503s (candidate/baseline 0.986020). Normalizing against unchanged attention-output, routed-down and shared-expert work gives aggregate ratio **0.985846**, and median matched-tail ratio **0.985016**. Both must be below 1.0 to select further timing. The raw gate/up saving is only about 0.20% of baseline summed stage GPU time, so this change alone cannot materially close the reference gap. 209 of 287 matched tails improve on that diagnostic. Median absolute deviation is 0.015313; all per-tail values are retained in `audit.json`.

## Scope and correctness

Only native 4K continuation-tail fused IQ2 gate/up grid order changes. `build.rs` appends a Rust-owned shader to the existing MoE library. The shared tail lazily creates/selects its pipeline under the existing `native_chunk` predicate and swaps grid axes. Initial prefill, legacy tiles, decode and imported root Metal are unchanged. Reversing two coordinate expressions and the function rename restores the baseline kernel body byte for byte; the IQ2 template parameter remains 16. `implementation.patch`, both kernel sources and `verify.py` preserve/check that identity.

The pinned GGUF is `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` (86,720,111,488 bytes), oracle-v5 source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. Each run checks 129,280 prefill FP32 words and 128 full decode rows, **16,677,120 words per run; 50,031,360 total**. Tokens alone are insufficient. All immutable output targets remain unchanged. This does not extend the accepted internal-HC frontier.

| Run | Full-output C0 | Process elapsed (s) | Diagnostic peak RSS (GiB) |
| --- | --- | ---: | ---: |
| candidate-32k-output | PASS | 289.398 | 3.107 |
| baseline-32k-profiled | PASS | 273.302 | 3.111 |
| candidate-32k-profiled | PASS | 260.253 | 3.108 |

These process durations include conformance/startup work and are not headline timing. Diagnostic RSS omits full engine page warming and does not establish total model/Metal residency or a memory improvement. No timed samples were selected or discarded; these are the three declared corrected-candidate runs.

## Diagnostic attribution

Both profiles pass 287 ordered native-tail packets (chunks 4096 through 28672, layers 2 through 42), each with nine strictly increasing counter intervals: 2,583 intervals per run. The audit rejects missing/duplicate coverage, invalid counters and inconsistent parent attribution; it checks the unchanged bootstrap/prefix coverage as well. Encoder splitting and MTLFence serialization perturb execution. Attributed component times distribute parent GPU duration according to counter deltas; they are not independent end-to-end stopwatches.

| Tail component | Baseline GPU attribution (s) | Candidate (s) | Candidate/baseline |
| --- | ---: | ---: | ---: |
| attention_output | 32.370772 | 32.374998 | 1.000131 |
| ffn_ingress | 0.714546 | 0.707066 | 0.989532 |
| router | 0.260291 | 0.262990 | 1.010371 |
| routed_map | 0.322521 | 0.322705 | 1.000570 |
| routed_gate_up | 30.856877 | 30.425503 | 0.986020 |
| routed_down | 16.552847 | 16.558326 | 1.000331 |
| routed_sum | 0.195658 | 0.195898 | 1.001228 |
| shared_expert | 4.969659 | 4.969500 | 0.999968 |
| final_hc | 1.104166 | 1.104014 | 0.999862 |

| Stage | Calls | Baseline GPU (s) | Candidate GPU (s) | Candidate/baseline |
| --- | ---: | ---: | ---: | ---: |
| initial_bootstrap | 1 | 5.144443 | 5.152133 | 1.001495 |
| prefix_layer | 40 | 18.417952 | 18.413263 | 0.999745 |
| bootstrap_ingress | 7 | 32.932848 | 32.946824 | 1.000424 |
| bind | 287 | 0.000000 | 0.000000 | n/a |
| boundary | 147 | 5.921824 | 5.909292 | 0.997884 |
| host_scratch | 147 | 0.000000 | 0.000000 | n/a |
| attention | 287 | 55.112393 | 55.102809 | 0.999826 |
| tail | 287 | 87.347338 | 86.921002 | 0.995119 |
| ingress | 140 | 6.870315 | 6.795991 | 0.989182 |
| compressor | 140 | 0.411700 | 0.415980 | 1.010395 |

The stage tables retain absolute changes in untouched work so ordinary run variation remains visible. The aggregate ratio uses sums of the three named controls; the matched statistic independently normalizes every corresponding tail. The experiment was sequential baseline then candidate, without alternating repeats; it supplies no confidence interval or production speed guarantee. The initial standalone synthetic screen is separate evidence in `../prefill-route-order-d6c14fa0/`.

## Build, failure and provenance

The first candidate `9171518a` failed library compilation before inference: appended source referenced `QK_NL` after the imported file undefines it. The failed run, source, build metadata, plan and host observations remain under `rejected/`. No profiles ran on that failed candidate. Correction `45520435` substitutes the original constant 16 in two template arguments; it changes no arithmetic or acceptance criterion.

All 318 release Rust tests and formatting passed on the first candidate. Their host-test evidence is explicitly reused for the two-constant Metal correction (`reused-host-tests.json`). The corrected release executable builds; the actual assembled Metal library and new pipeline compile on M1 Ultra (SIMD width 32, maximum 512 threads). All live C0/profile results above belong to that corrected frozen executable. Final verification rehashes the complete model, prompt, oracle source/executables, frozen Rust baseline/candidate and all 40 source inputs.

The sequential runner clears DS4/Rust-Star/Metal override variables, checks frozen source/executable hashes before each run and rejects concurrent model processes. It records host load, thermal status, VM state and swap every 30 seconds. There are 34 host observations; reported swap usage ranges 0.00–0.00 MiB. Raw observations remain available; this is not proof of exclusive system load.

## Reproduction

Run `python3 verify.py` here for model-free artifact hashes, source/build/input bindings, exact-output receipts and independent profile/statistics recomputation. This verifies retained evidence; it does not rerun the GPU. `artifacts.json` hashes every other report file. `redactions.json` binds original and stored digests for source work artifacts; local absolute paths are replaced with `<repo>`, `<model>` and `<user>`. The portable audit authenticates stored files before using their original digests. Binaries and weights are excluded.

Live reproduction requires the exact target machine/toolchain, pinned source/build and model. Build the runtime from the candidate commit and run sequentially:

```sh
"$CANDIDATE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json candidate-32k-output.json
"$BASELINE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json baseline-32k-profiled.json --profile-json baseline-32k-profiled-components.json
"$CANDIDATE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json candidate-32k-profiled.json --profile-json candidate-32k-profiled-components.json
```

`validate.py` is the archived original runner with redacted local paths, not a turnkey rerun script. Supply paths and the same environment/input checks when reproducing. The preserved assembled library can be checked with `compile-shader.m` using Foundation and Metal frameworks.

The next candidate-specific step is a bounded alternating unprofiled 32K engine timing comparison against the frozen Rust baseline, with an end-to-end prefill criterion fixed before running. Do not automatically start a larger campaign. No push or release is part of this research validation.
