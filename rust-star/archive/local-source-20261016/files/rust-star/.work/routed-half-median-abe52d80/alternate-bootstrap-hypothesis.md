# Possible next hypothesis if this routed study is not selected

This is source-only planning, not an executable plan or permission to interrupt
the current fixed study. Close the current result first.

The rejected 4K-bootstrap report shows a large isolated bootstrap saving but a
52.484% increase in macOS peak memory footprint (roughly 6 GiB), with higher
aggregate setup/host/remaining-command costs. A specific allocation/stall cause
has not been proved. A bounded 1K bootstrap batch is a distinct working-set
hypothesis: four calls per native chunk instead of one 4K call or sixty-four
64-row calls. It may keep much of the dispatch benefit with smaller buffers;
this remains unmeasured and may also fail.

If justified after the current result, freeze a separate 32K-only plan against
the then-current accepted source and same-compiler executable. Preserve the
original 4K rejection; do not rerun/relabel its pairs. Retain prefix/legacy/decode
and >32K behavior. Reuse the proven router group bound, but include 1K rows in
an actual-kernel synthetic dispatch proof. The new source must have complete
inverse/scope proof and fresh internal/full C0 before any performance decision.

Report peak RSS and separate peak memory footprint, all passive timing phases,
and the complete whole-engine block. An observed component gain or a lower
footprint alone cannot justify retention. No automatic 256K work or publication.
