from pathlib import Path
root=Path(__file__).resolve().parents[3]
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();i=s.index('## 2026-09-07 — Five optimized pairs');entry='''## 2026-09-07 — Prefill optimization milestone complete

- Profiled all native32K continuation layers/stages, identified the dominant
  even-layer tail, and replaced 18,816 32-row tail calls with 147 4K calls.
  Arithmetic kernels and accepted oracle-v4 are unchanged. Source candidate
  `dd0d26c5424acd3e8b27606771c46ed3c71e715a`; executable SHA-256
  `5d05eb14b3288f2233095e869a8f422dc9630f71484f7580ea482f3772e74b56`.
- All live gates pass: internal12K, two complete32K full-logit/GPU-selection
  modes, exact 2K/8K, two retained32K reset/reuse cycles, 310 Rust tests and
  91 Python tests. Internal HC acceptance still ends at 24K.
- Completed five valid AB/BA/AB/BA/AB pairs and both warmups. Confirmed host
  restart interrupted pair 5 attempt 1; retained it and reran both engines
  under the external-interruption protocol. No performance-based exclusion.
- Independently audited median prefill: Rust 521.130 ± 21.751 MAD s versus
  reference 184.333 ± 5.872 s. Prior Rust median 813.107 s: 35.91% latency
  reduction / 1.560x throughput across campaigns; reference-normalized gain
  1.521x. Remaining latency gap 2.827x. First four paired throughput ratios
  median 0.34065; post-restart pair 0.35696; all-five median 0.35133.
- Steady decode: 19.752 ± 0.201 versus 19.595 ± 0.363 tokens/s, paired median
  1.00368 (range 0.96658–1.05343): near parity, no robust decode gain.
  Median peak process memory 87.585 versus 80.893 GiB; prior Rust 91.645 GiB.
- Separate before/after profiles reconcile 1,483 rows each. Even-tail GPU
  time falls 294.625 to 43.453 s. Largest remaining stage: even attention,
  77.645 s GPU / 124.838 s host over 18,816 calls. Profiles are excluded
  from speed samples; grouped bootstrap coverage is explicit.
- Durable report: `rust-star/reports/32k-prefill-dd0d26c5/README.md`.
  Portable verifier passes all 106 artifacts, 12 accepted measurement bundles,
  the interrupted pair's completed reference bundle, exact transcripts, both
  profile reconciliations and independent statistics. 116 host observations
  contain zero recorded thermal/performance warnings; restart caveat retained.
- Next milestone: investigate even-layer attention batching and command/setup
  costs, preserve C0 and lifecycle gates, then repeat the paired protocol.
  This completes the authorized milestone, not the 256K project headline.
  Journal, charter and runtime docs updated. No push or packaged release.

''';s=s[:i]+entry+s[i:]
a=s.index('### Current resume checkpoint');b=s.index('### Accumulated implementation',a)
s=s[:a]+'''### Current resume checkpoint (updated 2026-09-07)

- Completed milestone: measured native32K prefill tail batching, candidate
  `dd0d26c5`, unchanged accepted oracle-v4 `df99eba7`. Durable report:
  `rust-star/reports/32k-prefill-dd0d26c5/README.md`; 106-artifact portable
  verifier passes. No model process remains active after final validation.
- Median prefill 521.130 s (previous Rust 813.107 s), 35.91% less latency;
  reference 184.333 s, leaving a 2.827x gap. Reference-normalized improvement
  1.521x. Decode remains near parity: 19.752 / 19.595 tokens/s. Median peak
  process memory 87.585 / 80.893 GiB. These are 32K development results.
- Five valid pairs and two warmups pass. Host restart interrupted pair 5;
  attempt retained and complete pair rerun under protocol. Report explicitly
  compares pre-/post-restart observations. Do not describe as uninterrupted.
- Profiled even-tail GPU time falls 294.625 to 43.453 s; next bottleneck is
  even attention (77.645 GPU / 124.838 host s, 18,816 calls). Investigate
  batching/command costs next, with exact validation and a new frozen plan.
- All 310 Rust and 91 Python tests, two fresh full32K C0 modes, internal12K,
  2K/8K controls and two retained32K reset/reuse cycles pass. Native32K is
  complete through all 43 layers, output head and 128 greedy steps. Do not
  restart completed layer work. Internal HC is accepted through 24K only;
  later internal hashes are diagnostic, never substitutes for full logits.
- Candidate executable SHA-256:
  `5d05eb14b3288f2233095e869a8f422dc9630f71484f7580ea482f3772e74b56`.
  Frozen plan `a446a6be765efa8198aec8b6111803b2b998dd3f98926b3ffd6341dae69859db`.
  Work: `rust-star/.work/paired-32k-128-dd0d26c5-batch/` and
  `rust-star/.work/prefill-milestone-fe73defe/`.
- Prior audited baseline: `rust-star/reports/32k-baseline-cbb078bb/`.
  Streaming top-k count race remains repaired with adversarial GPU coverage;
  original targets and failed projection-wait experiment remain preserved.
  Never resume rejected `paired-32k-128-7406473` or its `-sampling` sibling.
- Work in `/Users/deathcodevision/dev/ds4-rust-star-bootstrap` on
  `agent/rust-star-bootstrap`. Remotes: `fork` = `deathcoder/ds4`,
  `origin` = `antirez/ds4`. No push performed. Research crate is publish=false;
  current repository is not a Dion Labs packaged-release project.
- Use accepted oracle-v4 for new evidence; preserve historical v1/v3 fixture
  producer labels. Preserve GPU-owned histories, tied-score router ordering,
  shorter controls and reset/reuse. One large model process at a time.
- Reclaimed 10.55 GiB of duplicate build products this milestone; about
  131 GiB was available before host restart. Preserve frozen evidence and
  use `rust-star/.work/runtime-target` explicitly for future builds.
- Broader native contexts, the 256K headline and coding-agent co-design remain
  future work. This checkpoint supersedes older accumulated frontiers below.

'''+s[b:]
a=s.index('## Immediate Next Actions');b=s.index('## Entries',a)
s=s[:a]+'''## Immediate Next Actions

- The authorized prefill optimization milestone is complete. Report and
  verifier: `rust-star/reports/32k-prefill-dd0d26c5/`.

1. Next milestone: investigate native32K even-layer attention, now the largest
   measured stage (77.645 s GPU / 124.838 s host, 18,816 calls). Attribute
   batching, encoding/waits and allocation costs before choosing a change;
   validate a C0-safe improvement with a fresh frozen five-pair comparison.
2. Preserve exact 2K/8K controls, complete32K immutable output trace, GPU
   selection, GPU-owned history, reset/reuse and tied-score router ordering.
   Internal HC acceptance remains through24K. Do not redo completed layers.
3. Use accepted oracle-v4 for new captures. Preserve original fixture producer
   identities and all failed/interrupted attempts. Never resume rejected plans.
4. Broader native contexts and the 256K primary headline remain future work.
   Run one large model process at a time and keep disk use bounded.
5. Update the journal at each meaningful step and commit it with the work.
6. GitHub Actions billing lock and deferred secure remote access remain in
   the manual task ledger.

'''+s[b:];p.write_text(s)
p=root/'RUST_STAR_PROJECT.md';s=p.read_text();a=s.index('- Fresh and retained 32K engine validation');b=s.index('- Internal HC targets',a);s=s[:a]+'''- Fresh and retained32K validation, full-logit gates, 2K/8K controls and
  measured prefill tail batching are complete. Audited five-pair report:
  `rust-star/reports/32k-prefill-dd0d26c5/README.md`. Median Rust prefill
  falls 813.107 to 521.130 s (35.91% less latency across campaigns), versus
  reference 184.333 s. Decode remains near parity (19.752 / 19.595 tokens/s).
  A host restart interrupted the last pair; complete-pair rerun and original
  evidence are disclosed. Next: even-layer attention, the largest remaining
  measured stage, under the same exactness and paired-measurement protocol.
'''+s[b:];s=s.replace('As of 2026-09-06, native 32K','As of 2026-09-07, native 32K');p.write_text(s)
p=root/'rust-star/runtime/README.md';s=p.read_text();a=s.index('Fresh32K normalized measurement and both retained32K');b=s.index('At 32K, `engine-profile`',a);s=s[:a]+'''Fresh32K normalized measurement, both retained32K reset/reuse cycles and
full 2K/8K controls pass. Native32K even-layer tails now run once per 4K
chunk after retaining the existing 32-token attention tiles. All full-output
C0 gates pass; arithmetic kernels and oracle-v4 remain unchanged.

The audited [prefill report](../reports/32k-prefill-dd0d26c5/README.md) records
five complete pairs: median Rust prefill 521.130 s versus 184.333 s reference,
down 35.91% from the [earlier baseline](../reports/32k-baseline-cbb078bb/README.md).
Decode remains near parity (19.752 / 19.595 tokens/s). A host restart required
a complete final-pair rerun, with interrupted evidence retained. Even-layer
attention is now the largest measured stage and the next optimization target.

'''+s[b:];p.write_text(s)
