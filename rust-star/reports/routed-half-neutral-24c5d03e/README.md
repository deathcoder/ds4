# Routed-input half: exact outputs, invalid timing screen

Decision: **do not select or reject the kernel for speed**. The contemporary A/A screen fails in five of six case/session combinations, so the runner stops before candidate timing. Runtime source remains unchanged at `24c5d03e`.

The candidate moves repeated FP32-to-FP16 input conversion out of the fused routed gate/up kernel into one conversion dispatch. All IQ2 dequantization, MMA, clamp, SwiGLU and output operations remain unchanged; the complete source inverse and production arguments/encoding verify. Timing covers the whole map/conversion/gate-up chain. The fresh profile identified gate/up as 30.345s and 17.20% of recorded GPU stage time; this motivates the hypothesis without establishing an engine gain.

## Exactness and fixed protocol

All 16 edge cases pass at 1/31/32/33/63/64/65/1024 rows with two routing/input patterns: 64,536,576 FP16 words, exact CPU half conversion and route maps, finite/nondegenerate outputs, guards and unchanged inputs. Two fresh A/A sessions reverse three full 4096-row cases. They preserve all 72 pairs and compare another 7,851,737,088 FP16 words, for **7,916,273,664** total. Of that total, 668,516,352 word comparisons directly exercise candidate-versus-original outputs in edge checks and full-case warm-ups; the rest are A/A reference checks. These are synthetic intermediate outputs, not full-model C0.

Each role uses the same active buffers and the same original pipeline object. Four complete conditioning chains precede three measured chains; their arithmetic mean is the slot value. Every role is checked against a separate original-kernel reference, and active arenas alternate. All conditioning and measured command intervals are preserved. No samples, pairs or sessions are excluded.

| Session | Case | First/second median | Ratio MAD | AB role median | BA role median | Pairs within 3% | Result |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0.998185 | 0.013854 | 1.004947 | 0.999738 | 8/12 | FAIL |
| 0 | 1 | 0.996089 | 0.003751 | 1.004370 | 0.996208 | 11/12 | PASS |
| 0 | 2 | 1.004586 | 0.007810 | 0.996062 | 1.011484 | 11/12 | FAIL |
| 1 | 2 | 0.997876 | 0.012812 | 1.000722 | 0.990833 | 9/12 | FAIL |
| 1 | 1 | 1.008721 | 0.016097 | 0.978077 | 1.000584 | 7/12 | FAIL |
| 1 | 0 | 0.998813 | 0.018621 | 1.007624 | 1.017435 | 9/12 | FAIL |

Neutral criteria were fixed beforehand: first/second, separate AB/BA and arena medians within 1%; ratio MAD at most 1%; at least 10/12 pairs within 3%. Every cell must pass. Candidate selection would additionally require every case/session/order median below 0.98 and overall median below 0.95. Those candidate measurements never ran.

## Diagnosis and next experiment

The descriptive raw-triplet audit finds a maximum within-slot duration ratio of 1.457587; one measured chain can be about 46% longer than another in the same three-sample slot. Median triplet coefficients of variation range from 0.0054 to 0.0105. This does not identify a specific scheduling or hardware cause.

A separate prospective protocol can test longer conditioning and a longer median-based sample window while retaining the same neutral and candidate thresholds. Preserve this result as failed; do not recompute it with the new estimator and call it passing. A new protocol needs fresh neutral evidence before any speed claim.

## Verification

Run `python3 rust-star/reports/routed-half-neutral-24c5d03e/verify-report.py`. It verifies artifact/source/binary identities, the complete kernel inverse and Metal assembly, ABI structs, unchanged original/current host fragments, edge checks and every raw A/A command/decision. It reruns the frozen audits and descriptive variance calculation with Python only; it does not run the GPU. The prior wide-bootstrap report must remain beside it for the complete host snapshots. Private workspace paths are redacted with separate original/stored hashes; the probe executable is omitted.

No runtime integration, model run, candidate performance result, DwarfStar comparison or publication occurred. The earlier 256K campaign remains paused. If a later local protocol selects this kernel, current-baseline full C0 and a separately frozen whole-engine study are still required.
