# Routed-input half with a longer median-based protocol

Decision: **do not select under this local study**. This is a standalone synthetic-kernel result; the runtime remains unchanged and no engine gain is established.

## Protocol and provenance

The preceding mean-of-three protocol failed A/A neutrality and remains a closed failure. This new attempt fixed eight conditioning chains, twelve measured chains and their median as the primary slot estimate before compilation. Every raw interval and its arithmetic mean remain recorded. Kernel source, complete Metal assembly, input cases/seeds, pair orders, arena schedule and every neutral/candidate threshold are unchanged. The complete harness/audit/runner edit ledger reverses byte-for-byte to the prior protocol.

The candidate converts normalized FFN input to FP16 once before the existing routed fused gate/up calculation. Timing includes route-map, conversion and gate/up work. Every IQ2 dequantization, MMA, clamp, SwiGLU and output operation is preserved. Source/ABI/production-host proofs, finite outputs, CPU map/half checks, guards and unchanged inputs all verify.

## Exactness and timing controls

All 16 fresh edge cases pass. The complete study checks 23,015,768,064 FP16 output words; 4,896,374,784 of those directly compare candidate and original outputs in edge checks, full-case warm-ups and candidate roles. The remainder are A/A reference checks. These are repeated synthetic intermediate comparisons, not full-model C0.

Initial contemporary A/A screen: PASS. Both sessions reverse three 4096-row cases and retain all 72 pairs. The two mixed sessions retain another 144 pairs, interleaving 72 candidate comparisons with 72 A/A controls; mixed neutrality fails.

| Phase | Session | Case | First/second median | MAD | AB ratio median | BA ratio median | Within 3% | Neutral gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| initial | 0 | 0 | 0.995478 | 0.003687 | 0.998863 | 0.994612 | 12/12 | PASS |
| initial | 0 | 1 | 1.001695 | 0.003694 | 1.001831 | 1.002691 | 12/12 | PASS |
| initial | 0 | 2 | 1.000750 | 0.004229 | 0.994980 | 0.997539 | 12/12 | PASS |
| initial | 1 | 2 | 1.001875 | 0.001836 | 0.999682 | 1.003404 | 12/12 | PASS |
| initial | 1 | 1 | 1.002409 | 0.004272 | 0.997028 | 0.995501 | 12/12 | PASS |
| initial | 1 | 0 | 0.995782 | 0.006200 | 1.006979 | 1.000061 | 11/12 | PASS |
| mixed | 0 | 0 | 1.000510 | 0.005457 | 1.000058 | 1.000510 | 12/12 | PASS |
| mixed | 0 | 1 | 0.998904 | 0.002859 | 1.004591 | 0.999083 | 12/12 | PASS |
| mixed | 0 | 2 | 0.993674 | 0.005796 | 1.004172 | 0.989395 | 12/12 | FAIL |
| mixed | 1 | 2 | 1.005183 | 0.009834 | 0.990703 | 0.991805 | 12/12 | FAIL |
| mixed | 1 | 1 | 0.996399 | 0.006463 | 1.005761 | 0.998025 | 12/12 | PASS |
| mixed | 1 | 0 | 0.995983 | 0.007000 | 1.004036 | 0.999812 | 12/12 | PASS |

Neutral criteria require every cell to pass: first/second, AB/BA and arena medians within 1%, ratio MAD at most 1%, and at least 10/12 pairs within 3%. No command, pair or session is excluded. The prior failed study is not recalculated or relabeled.

## Descriptive candidate samples

Mixed neutrality failed: these samples do not establish a reliable kernel speed gain. The candidate also misses the unchanged selection thresholds. Descriptive overall median paired ratio: **0.984955** (-1.505% whole-chain GPU latency). Each role is the median of twelve measured complete-chain intervals. The primary paired ratio is distinct from the ratio of aggregate role medians.

| Session | Case | Baseline median ms | Candidate median ms | Paired ratio median | AB median | BA median | Speed gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 89.839063 | 87.917521 | 0.982167 | 0.980800 | 0.983891 | FAIL |
| 0 | 1 | 88.854010 | 87.529729 | 0.985322 | 0.984601 | 0.987199 | FAIL |
| 0 | 2 | 89.594375 | 87.594365 | 0.977074 | 0.970733 | 0.985245 | FAIL |
| 1 | 2 | 90.036542 | 88.391146 | 0.983297 | 0.985349 | 0.979475 | FAIL |
| 1 | 1 | 89.505719 | 88.004844 | 0.985531 | 0.986591 | 0.982963 | FAIL |
| 1 | 0 | 84.669740 | 83.501781 | 0.988566 | 0.989478 | 0.987987 | FAIL |

Candidate criteria require every case/session/order median below 0.98, overall paired median below 0.95, and all exactness/contemporary neutral gates passing. Thresholds were fixed before execution and unchanged from the prior study. The full raw result and every gate are in timing/audit.json.

## Verify and continue

Run `python3 rust-star/reports/routed-half-median-abe52d80/verify-report.py`. This Python-only verifier checks every artifact/source/binary identity, complete kernel assembly/ABI/host proofs, the complete protocol inverse and unchanged thresholds, then reruns the frozen edge and timing audits against all raw commands. It does not use the GPU or model. Keep the previous routed-half-neutral and wide-bootstrap reports beside this report for their immutable source snapshots. Private paths are redacted with original/stored hashes; the executable is omitted.

All 42 runtime source hashes remain unchanged. Close this local result and continue the next justified hypothesis. A failed speed criterion must not be rescued by excluding cases or adjusting thresholds.

No fresh full-model, engine, DwarfStar or higher-frontier claim is made. No publication occurred and the interrupted 256K campaign remains paused.
