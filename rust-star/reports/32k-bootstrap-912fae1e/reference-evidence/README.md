# Rust Star: dense Q8 output projection — not retained

The experiment passed its exact-output gates but did not meet the measured improvement criterion. The dense-output implementation is reverted to the accepted grouped projection path. The component profiler and its matching C declarations remain. No new performance baseline is accepted.

Native tails were changed to eight aligned dense-Q8 output-low dispatches with direct token-major stores, using a physical pitch of 8192. This removed the grouped map/scatter path and two scratch buffers (22 → 20), while increasing native dispatches from 25 to 31. Metal kernel arithmetic was unchanged. `implementation.patch` preserves the rejected change.

## Five-pair evidence

Five alternating AB/BA pairs completed, with one prior warmup per engine. Correctness collection and profiling were disabled for timed runs; initial selection belongs to prefill, and decode includes evaluation, selection and commitment. Each engine invocation was a fresh process. These are 32K/128 development results.

| Distribution (5 valid runs each) | Median | MAD | Min–max |
|---|---:|---:|---:|
| Rust prefill seconds | 351.974 | 19.389 | 332.585–424.092 |
| Rust steady decode tokens/s | 20.170 | 0.391 | 16.825–20.878 |
| Rust first decode ms | 58.152 | 4.819 | 53.334–361.839 |
| Oracle-v5 prefill seconds | 176.897 | 2.765 | 172.088–207.320 |
| Oracle-v5 steady decode tokens/s | 19.821 | 0.150 | 17.682–20.225 |
| Oracle-v5 first decode ms | 51.783 | 1.912 | 49.871–59.696 |

| Pair | Attempt | Order | Rust prefill s | Oracle prefill s | Rust/oracle latency |
|---|---:|---|---:|---:|---:|
| 1 | 1 | AB | 351.974 | 207.320 | 1.69773 |
| 2 | 1 | BA | 332.585 | 176.897 | 1.88010 |
| 3 | 1 | AB | 424.092 | 177.746 | 2.38594 |
| 4 | 2 | BA | 373.952 | 174.132 | 2.14752 |
| 5 | 1 | AB | 346.732 | 172.088 | 2.01485 |

Median paired prefill latency ratio: **2.01485**, versus **1.72589** for the accepted native-tail baseline (+16.74% ratio change; lower is better). Paired steady-decode throughput ratio: 1.01664.

- Candidate median prefill: 362.284 → 351.974 s (-2.85%).
- Oracle median prefill: 212.376 → 176.897 s (-16.71%).

The absolute Rust reduction does not establish a benefit from this optimization: the reference improved more across the campaigns. These runs do not isolate the code change from varying host conditions. The criterion requires both lower absolute median prefill and a lower paired latency ratio; it failed.

## Interruption and host conditions

Pair 4 attempt 1 was interrupted during its oracle half by a host shutdown/restart. Its completed Rust half (538.625 s prefill) and original state remain preserved. The documented external-event recovery procedure reran the entire BA pair as attempt 2. The other completed pairs, including the slower third pair, were not discarded or retried. Raw evidence contains all six attempts; statistics use the five valid complete pairs. Recovery boot/state evidence is under `inputs/`.

The campaign spans the restart, with warmups preceding it. Background browser, video/camera, updater and other host activity varied. Source review, report-helper edits and a temporary-copy Rustfmt check of about six seconds during pair 3 also occurred. No runtime build or second model process ran alongside timed measurements. Full host/load observations are included.

The audit retains 1 unusual thermal observations, including shutdown-related query failures. Median peak process memory was 91.483 GiB for Rust and 80.894 GiB for the reference. These are observed residency values under varying conditions, not allocation-capacity claims.

## Correctness and validation

- 312 Rust tests and the candidate release build passed. The unchanged Python toolkit retains its 93-test acceptance evidence.
- All seven live gates passed: native 12K internal-state control, full 8K legacy control, 2K control, complete 32K output-only and internal modes, retained two-cycle reset/reuse, and a separate profiled full-output 32K gate.
- Both complete 32K modes and the profiled full-output gate match all 129,280 prefill logits and all 16,547,840 FP32 words across 128 decode steps. Accepted internal HC targets through 24K match. The 12K gate does not claim full 12K output-tensor equivalence.
- Retained lifecycle verifies both oracle transcripts; its same-process timings are ineligible for the paired comparison.
- Oracle-v5, fixtures, tokenizer evidence and historical failures are preserved. No tolerance or oracle change was introduced.
- Independent auditing rehashes every valid measurement, both warmups, the interrupted candidate measurement and its artifacts, and reproduces all paired statistics.

## Separate profile failure

The post-paired `engine-profile` attempt exited 1 with `rust-star: invalid tail timestamps` and wrote no profile artifact. The original log and explicit failure record are preserved as `profile32k.log` and `profile-failure.json`. It was not retried. The audit verifies this failure; it does not claim a successful post-paired profile.

The getter rejects an error-valued counter or any non-increasing adjacent timestamp. Rejected raw values were not logged, so the exact condition is unresolved. Keep this fail-closed behavior until better diagnostics establish the cause; no timestamp-validity condition was relaxed. This is a known limitation of the optional profiler. Normal timed inference completed all five valid pairs.

## Earlier exact-gated component evidence

The separate premeasurement profiled C0 gate succeeded. Its 287 tails and 2,583 component rows pass ordering, coverage and attribution conservation checks. This evidence is distinct from the later failed profile. Counter shares are scaled to parent command GPU intervals and are perturbed attribution, not standalone kernel latency or a throughput result.

| Component | Attributed GPU seconds | Share |
|---|---:|---:|
| attention_output | 35.248 | 36.08% |
| routed_gate_up | 34.254 | 35.07% |
| routed_down | 19.377 | 19.84% |
| shared_expert | 5.808 | 5.95% |
| final_hc | 1.307 | 1.34% |
| ffn_ingress | 0.781 | 0.80% |
| routed_map | 0.334 | 0.34% |
| router | 0.301 | 0.31% |
| routed_sum | 0.277 | 0.28% |

## Disposition and reproduction

The dense-output code is reverted; both changed execution files exactly match source 9c1d5009. Fresh 312-test and release-build checks pass; their logs, restored source hashes, rebuilt executable hash and restoration patch are included. These checks supplement the earlier full-C0 evidence for that restored implementation; no new live gate or performance claim is implied. The validated grouped path and optional component profiler remain. Improve invalid-counter diagnostics before relying on unattended profiling. The proposed layers-0..2 initial-bootstrap scope and broader-context allocation work remain unapplied follow-up work. Stop at this milestone as requested.

Rejected candidate source: `92ca3ef3ee21ea076852d7ecbda415a15acb270a`. Executable SHA: `f2a0679725b9c21e0946561d42920b1afa82f05208aa66426268b7ecfef13ea9`.
Oracle source: `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. Executable SHA: `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`.
Frozen plan SHA: `c046b3e84019aca1e25e8254930743c04f1d250111b17ef779ff6929baf15590`.

Run `python3 verify.py` without the model to verify artifact integrity, paired statistics, the explicit interrupted attempt, full-output evidence, premeasurement component attribution and the preserved post-paired failure. The accepted performance baseline remains [32k-tail-89e32745](../32k-tail-89e32745/README.md); the original component study remains [32k-tail-components-e7b81aa2](../32k-tail-components-e7b81aa2/README.md).
