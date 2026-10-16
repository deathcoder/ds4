# Next milestone: explicit bounded scratch reuse

The checked wait limits unfinished work, but did not reliably make old scratch
objects disappear before the next allocation. A new candidate should establish
its storage bound through explicit ownership rather than prompt destruction.
This proposal is not implemented and does not reopen the failed study.

1. Start with a finite, explicitly owned scratch slot for the eligible native 1K
   path. Define its label-to-buffer map, capacity and lifecycle. Distinguish all
   persistent KV/compressor/output state from reusable scratch. Avoid aliases
   between different logical scratch names.
2. Keep the checked completion boundary before CPU initialization or reuse.
   Prove that every selected buffer is fully written before its next read, or
   give it explicit initialization matching its contract. The 105-name lexical
   inventory alone does not establish this. Reject unsafe pooling choices.
3. Define allocation/identity checks prospectively: at most the declared slot
   count and capacity per scratch label, completed prior GPU use before slot
   writes, and correct release/reset of the owner. An intentionally owned pool
   cannot use the old after_pool==zero criterion. This is a different ownership
   design, not a reclassification of the current candidate as passing.
4. Preserve the initial prefix, 32-row diagnostics, 64-row callers, decode,
   higher-frontier paths, shader arithmetic, immutable oracle and fixtures.
5. Freeze a new plan, source/binaries and scripts. Require local source/write
   proofs, fresh internal/full-C0 checks and a pool-aware lifetime diagnostic
   before any engine timing. Keep the established fixed engine pairs and every
   performance/RSS/footprint threshold, including footprint median ratio <=1.15.

If the ownership or first-write proof remains incomplete, report that result
without a model benchmark. Do not assume that releasing a command reference,
adding another wait or sleeping guarantees destruction of every resource; such
claims need independent evidence. No publication or 256K work belongs here.
