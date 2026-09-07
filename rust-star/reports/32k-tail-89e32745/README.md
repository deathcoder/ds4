# Rust Star: native tail scratch reuse and direct output

Native 4K tails reuse 22 command-local scratch buffers through the existing prefill transient cache. Retained HC outputs stay separate, and the final HC kernel writes directly into its full-chunk destination. The unused native routed-mid FP32 conversion and its 192 MiB destination are removed. Kernel arithmetic and legacy diagnostic paths are unchanged. The shared executor applies these changes to both even and odd native tails.

## Five-pair result

Five alternating AB/BA pairs and one warmup per engine passed on their first attempt. Initial selection belongs to prefill; decode timing includes evaluation, selection and commitment. Correctness collection and profiling are disabled during the paired block. These are 32K/128 development results.

| Metric | Rust median | Oracle-v5 median |
|---|---:|---:|
| Prefill | 362.284 s | 212.376 s |
| Steady decode | 17.064 tokens/s | 17.039 tokens/s |
| First decode step | 68.113 ms | 60.158 ms |
| Peak process memory | 91.613 GiB | 80.895 GiB |

| Pair | Order | Rust prefill s | Reference prefill s | Rust/reference |
|---|---|---:|---:|---:|
| 1 | AB | 361.131 | 212.376 | 1.70043 |
| 2 | BA | 362.284 | 213.088 | 1.70016 |
| 3 | AB | 371.311 | 212.898 | 1.74408 |
| 4 | BA | 362.122 | 209.817 | 1.72589 |
| 5 | AB | 376.401 | 210.228 | 1.79044 |

OBS/camera/video activity appears in the process observations from 2026-09-07T20:09:44.275156+00:00 through 2026-09-07T21:20:23.113142+00:00 (the observer also covers the separate profile). Background load varies across the observations. All pairs are retained; the exact latency improvement cannot be attributed solely to the code change under these varying background conditions.

Host swap before the campaign: `vm.swapusage: total = 4096.00M  used = 3098.88M  free = 997.12M  (encrypted)`; after the timed block: `vm.swapusage: total = 4096.00M  used = 2794.69M  free = 1301.31M  (encrypted)`. Peak process memory is an observed residency measure under these host conditions, not proof of a lower allocation requirement.

Paired Rust/reference prefill latency ratio: 1.72589. Steady decode ratio: 0.99758. Complete median/MAD/range distributions and independent recomputation are included.

## Change from the ingress-batching baseline

- Candidate median prefill: 371.601 → 362.284 s (-2.51% latency).
- Oracle median prefill: 206.136 → 212.376 s (+3.03% latency).
- Rust steady decode: 17.264 → 17.064 tokens/s; paired Rust/reference ratio 1.00776 → 0.99758.
- Reference-normalized prefill improvement across campaigns: 1.047× (paired latency ratio 1.80652 → 1.72589).
- Rust peak memory: 92.250 → 91.613 GiB.

Both campaigns use the same pinned oracle-v5, model, prompt, full-output targets and greedy transcript. They are separate host sessions; the normalized comparison is descriptive. Background applications remained active, and full load/thermal observations are included.
The audit records 0 unusual thermal observations.

## Validation

- 310 Rust tests pass. The unchanged Python toolkit retains its 93-test acceptance evidence.
- Fresh complete 32K output-only and diagnostic runs match every prefill logit and all 128 full decode outputs to oracle-v5. Every accepted internal HC target through 24K matches.
- The 12K native control, detailed 8K legacy control, 2K control and two retained reset/reuse cycles pass. No fixture was updated or tolerance introduced.
- The full v5 tensors, reference repeatability and tokenizer evidence are reused unchanged from the audited baseline report. Historical failed oracle/diagnostic attempts remain preserved and explicitly separate from this candidate.
- Existing decoder, prefix, SSD and distributed implementations are unchanged. The 32-row standalone controls retain their diagnostic conversion and output assembly. Native tails use 25 dispatches instead of 26.

## Isolated profile

Host elapsed time includes command waits and is not additive with GPU time.

| Stage group | GPU seconds | Host elapsed seconds | Calls |
|---|---:|---:|---:|
| even/tail | 51.665 | 59.400 | 147 |
| odd/tail | 49.172 | 56.772 | 140 |
| even/attention | 42.603 | 45.901 | 147 |
| bootstrap/bootstrap_ingress | 38.723 | 40.892 | 7 |
| prefix/initial_bootstrap | 29.353 | 29.613 | 1 |
| odd/attention | 28.867 | 32.701 | 140 |
| odd/ingress | 26.292 | 33.144 | 140 |
| even/boundary | 23.876 | 31.997 | 147 |
| prefix/prefix_layer | 21.869 | 27.680 | 40 |
| odd/compressor | 0.581 | 4.719 | 140 |
| even/bind | 0.000 | 2.960 | 147 |
| even/host_scratch | 0.000 | 1.209 | 147 |
| odd/bind | 0.000 | 0.301 | 140 |

Even-layer tails retain 147 stage calls; GPU time changes from 49.919 to 51.665 s and host time from 66.497 to 59.400 s. This separate instrumented run is ineligible for throughput claims.

Odd-layer tail host time likewise changes from 63.637 to 56.772 s. GPU time does not improve in this profile. The candidate is retained because exact outputs and lifecycle pass, paired prefill latency and its reference-relative ratio improve, and measured tail host time falls. These observations do not establish a precise isolated speedup under the different host conditions.

The next measured stage group is `even/tail`. Inspect its component costs before choosing another optimization, preserving full v5 C0 and lifecycle; track peak memory alongside speed. Broader contexts and the 256K headline workload remain future work.

## Reproduction

Candidate source `89e3274568745fe5311d16478ccf35f016958a08`, executable SHA `7f14080aea91a3631bb85b8c55dd8f02031e983ab2b74cc1618235a0262a74a4`.
Oracle source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, executable SHA `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`.
Frozen plan SHA `d27a9f5e56d6c804f62c10173a9a4e324498eba52d40227709b7d23cf15efe2f`.

Run `python3 verify.py` here without the model. It verifies all artifact hashes, raw paired statistics, transcripts, complete v5 tensors, candidate gates and profile coverage. `implementation.patch` records the runtime changes; `inputs/plan-template.json` and `portability.json` retain the frozen configuration and path-redaction provenance.
