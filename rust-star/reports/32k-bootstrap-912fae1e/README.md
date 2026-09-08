# Scoped initial bootstrap and counter diagnostics — 32K Rust study

**Bootstrap retained.** Prefill latency improved in all five pairs by 6.0–8.7%, with an 8.165% median paired reduction. Steady decode had a +0.516% median paired change (four pairs faster, one 2.0% slower), so there is no clear material steady-decode regression. First-step latency has a small, variable increase: the paired median is +2.121%, with observed candidate/baseline medians of 57.479/53.608ms. This tradeoff is accepted alongside the repeatable prefill improvement; no memory reduction is claimed.

The enhanced rejected-counter diagnostics are retained. All 312 Rust tests, the release build, all seven live gates and the component/probe audits pass. Five pairs and two warmups completed with no invalid attempts or retries.

| Metric | Candidate median | Accepted Rust median | Within-pair ratio median |
|---|---:|---:|---:|
| Prefill seconds | 273.573 | 298.948 | 0.91835 |
| Steady decode tokens/s | 20.557 | 20.384 | 1.00516 |
| First decode step ms | 57.479 | 53.608 | 1.02121 |

Prefill ratios below 1 are faster; decode throughput ratios above 1 are faster.

| Pair | Order | Candidate prefill s | Baseline prefill s | Latency ratio | Decode throughput ratio |
|---|---|---:|---:|---:|---:|
| 1 | AB | 273.194 | 294.871 | 0.92649 | 1.00731 |
| 2 | BA | 271.753 | 295.914 | 0.91835 | 1.01058 |
| 3 | AB | 273.608 | 298.948 | 0.91524 | 1.00234 |
| 4 | BA | 273.573 | 299.536 | 0.91332 | 1.00516 |
| 5 | AB | 285.216 | 303.413 | 0.94002 | 0.98001 |

Prefill dispersion (median absolute deviation; minimum–maximum): candidate 0.379s; 271.753–285.216s. baseline 3.034s; 294.871–303.413s. 

The separate post-paired profile passed transcript validation and interval accounting: 1,483 parent rows and 2,583 component rows. Its initial-bootstrap GPU span was 5.129s. These are diagnostic measurements, separate from the timed pairs.

Host observations: 85 thermal/swap samples reported no thermal warnings; swap used ranged from 13.94 to 15.12MiB. Desktop applications remained open. Timed samples included bursts from the renderer, Boxie and softwareupdated (up to roughly one CPU core); larger WindowServer bursts appeared during the separate profile. Logitech activity also appears in the retained logs. These are development measurements on this host, not an isolated laboratory or broader-context result. All host/load observations are preserved.

Observed process peak memory was 93.281GiB for the candidate versus 92.892GiB for the baseline maximum (baseline median 92.744GiB). No memory-reduction claim is made.

The final profile attributed 87.132s to native tails. Its largest components were attention output (32.957s), routed gate/up (30.155s), and routed down (16.493s). This leaves clear future profiling targets, but no additional optimization is included in this milestone. Both current profiles and the final probe passed; the earlier rare counter failure remains unexplained.

Candidate: 912fae1ebbd641fd8edaf4e9f6b26249737775b0, tree
8cb3b90da6fc288137efa13d1ef6eb4d84e671c2, executable SHA-256
adea36df8200ae6c99f4b549faebcf90fd01f9da34207dc0add562180cf713c4.

Comparator: accepted Rust restored at 94565861. Its executable SHA-256 is
a0ef134f35bc5b32df616951edba1b69ac4025c2fafe6642840b9699a1b2e75e,
identical to the full-C0-gated e7b81aa2 component-profiler build. The runner's
`oracle` slot holds this Rust comparator. Oracle-v5 remains the independent C0
reference, source ccbd16d36895ef9fe8edf7f793014988ff1ab7f4, executable
a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0.
This study does not update the Rust-versus-DwarfStar throughput result.

The initial 4K bootstrap now ends after layers 0–2. It retains layer-2 HC,
then uses the established exact prefix rebuild for layers 3–42. The former
bootstrap also executed those later layers and an output head whose results
were discarded. The new path omits that work. The complete 2K/legacy entry
points preserve their contract. No Metal kernel arithmetic or oracle targets
changed. Shared setup allocations and all 1,218 wrapped model views still occur;
this is not an allocation-removal or longer-context capacity claim.

Counter validity remains strict. Errors contain the chunk, layer, failing
sample/component/boundary, reason and all 18 raw hexadecimal ticks. Rust also
rejects the Metal error sentinel before attributing durations. The previous
failed profile did not preserve raw values, so its exact cause remains unknown.
The standalone probe uses nine dependent encoders, four widths and 64 repetitions
per width, checking GPU output while retaining all timestamp packets. The initial
probe preceded the output-exists guard; its evidence is retained separately.
The final probe's source, build and output are hash-bound by provenance.

Acceptance was declared before measurements: lower median within-pair prefill
latency, improvement in at least four of five alternating pairs, and review of
decode distributions for a clear material regression. Both roles have one warmup.
All attempts are retained; favorable retries are prohibited. Profiles are separate
runs and ineligible for headline timing. Counter component attribution scales
within-command spans to the completed command GPU duration; do not add the
components to their already-accounted parent rows.

Full output C0 means 129,280 prefill words plus 16,547,840 decode words, against
the pinned oracle-v5 fixtures. Accepted internal HC checkpoints end at 24K;
28K/32K internals remain diagnostic. The 12K internal control checks no full
output rows. The retained reset/reuse gate checks two complete token transcripts,
not two complete tensor traces. Timed measurements validate greedy transcripts
after each step and perform no full-logit correctness collection in the interval.

The bundle includes immutable copies of the previous negative study and accepted
component study so its reference and restored-baseline proofs can be verified
without the model, Metal, Git or neighboring reports. Their old measurements,
interrupted attempt and failed profile are historical evidence, not new runs.
The outer manifest includes both nested manifests. Frozen candidate sources are
included; binaries and the 86.7GB model are not included. Original absolute paths
in the frozen plan are provenance only and are not portable execution paths.

Verify with `python3 verify.py` from this directory. The independent audit
rehashes measurement artifacts for both Rust roles, recomputes intervals/rates,
checks complete transcripts and reproduces all distributions. The full verifier
also checks current C0 checksums against the bundled reference words, accepted
internal checkpoints, bootstrap/profile coverage and probe counters.
