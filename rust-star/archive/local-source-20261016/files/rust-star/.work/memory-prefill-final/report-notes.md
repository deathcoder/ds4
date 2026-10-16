## Change and evidence scope

Candidate `de6d2c11` combines allocation specialization (`8b511b5f`) and chunk-local bootstrap activations. The accepted comparator is `912fae1e`. Both use the same initial layers-0..2 bootstrap and exact prefix rebuild, native 25-dispatch tails, and oracle-v5 target. The generic paired-runner `oracle` slot holds accepted Rust; it is not the DwarfStar correctness oracle.

Partial bootstrap wraps 16 execution model ranges plus two indexer ranges, instead of 1,218 views, and omits discarded layers-3..42 allocations and initialization. Native continuation Q, HC, normalized-input and split storage is bounded to 4K rows. Producers, sparse/Flash consumers, tails and bootstrap HC checksums use the matching relative offsets. Full KV, compressed history and token storage retain absolute addressing. Execution-only Rust paths omit unused host diagnostic vectors. No kernel arithmetic or oracle fixture changed. Legacy controls retain their layout.

The preliminary allocation tranche has its own 312-test/build and 12K internal result in `phase1/`; it is not a full-output acceptance claim. The final candidate has seven gates in `correctness/`: native 12K internal, legacy 8K, complete 2K, 32K output-only, 32K internal, two retained cycles, and profiled 32K output. The three 32K gates compare all 129,280 prefill logits and all 16,547,840 FP32 decode words over 128 steps. Internal HC references are accepted only through 24K; the retained cycles validate transcripts, not two full tensor traces.

Oracle-v5 source is `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, executable SHA-256 `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`, on upstream `c0a6119f363ef82125877142f13fb3fe491cba14`. The selected prefill token is 85166, final decode token 14; prefill/decode-word/token FNV checksums are 2996662110144943749 / 1716816788919264169 / 11977321378026793466. Model and prompt are rehashed before pairing and pinned in `inputs/freeze.json`. This study does not change the latest accepted Rust-versus-DwarfStar result.

Five AB/BA alternating pairs and one warmup per role use fresh processes. The acceptance plan is hashed before the timed block: lower median peak RSS, lower RSS in at least four of five pairs, median paired prefill latency at most 1.03, and median paired steady decode throughput at least 0.97. Each measured transcript is checked. The component profile is separate and deliberately changes encoder boundaries; it is not paired performance evidence. Raw attempts, host observations and all timing/artifact identities are retained.

## 128K capacity limits

The standalone Metal probe allocates all 43 modeled full-KV histories, the 41 applicable compressed attention histories, 21 indexer histories, and explicit selected chunk transients simultaneously. It fills every buffer through a GPU blit and checks each 16K page plus the final byte. It records the live device buffer limit and recommended working set. It excludes model residency, other scratch and inference, and is not a complete 128K runtime proof.

`capacity-analysis.json` separates measured 32K peak RSS from a planning estimate that adds only the 32K-to-128K retained-history increase. That estimate is not an upper bound: grow-buffer overlap, other scratch, model-view alignment, driver overhead and decode handoff can add memory. Full KV still grows with context. The former per-slot 128K FullQ size of 16GiB is now 512MiB; per-slot HC falls from 8GiB to 256MiB. These are buffer calculations, not measured total-memory savings.

The runtime remains guarded at 32K. Next acceptance must pin repeatable 128K tokenization/oracle outputs, audit larger sparse selector schedules and decode adoption, then advance through intermediate frontiers and require full 128K prefill plus 128 full-logit decode steps under memory monitoring. Capacity evidence alone does not establish C0 or 128K throughput.

## Reproduction and retention

Run `python3 verify.py` from this report directory. The portable verifier checks every artifact hash, source snapshots, both measured roles, full-output fixture checksums, accepted HC boundaries, component intervals, allocation probe provenance and the recorded retention decision. It needs neither Metal nor the model. Accepted baseline provenance is copied separately; its full historical study remains at `../32k-bootstrap-912fae1e/`. The model and executables are excluded from this report.

The host was shared with interactive applications. Raw thermal/swap/load observations are retained; results are local development measurements rather than isolated-laboratory estimates. Previous negative experiments and the unexplained historical counter failure remain unchanged. Stop after this milestone.
