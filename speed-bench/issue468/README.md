# Issue 468 comparison corpus

These are byte-for-byte copies of the three 8k prompt fixtures used by
`run_mtp_verifier_bench_long.py` on lobanov/ds4's
`dspark-research/issue468` branch. `provenance.json` pins the source commit,
upstream token counts, byte sizes, and SHA-256 hashes.

`mtp_reference.json` freezes the published legacy MTP table used by the local
DSpark comparison report. It is reference data, not a claim that absolute t/s
is portable across machines. The MTP results were single instrumented runs;
the DSpark harness uses paired uninstrumented throughput samples and can run a
separate diagnostic pass.

The harness defaults to the exact DSpark target verifier. Its experimental
`--fast-verifier` mode is known to diverge on this corpus and is retained only
for correctness investigation. Do not report its throughput until the target
batch verifier preserves exact decode state across repeated commits.

The separate `--exact-head-batch` option retains exact target state and final
continuation logits while batching intermediate acceptance heads. It has passed
all three fixtures but has not yet been performance-benchmarked.

Do not edit the prompt files. Replace them only from a newly pinned upstream
commit and update all provenance fields and hashes together.
